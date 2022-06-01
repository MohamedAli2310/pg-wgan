"""
Application entry point for PG-GAN.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date 2/4/21
"""

# python imports
import datetime
import math
import numpy as np
import random
import sys
import tensorflow as tf
from scipy.optimize import basinhopping

# our imports
import discriminators
import real_data_random
import simulation
import util

from real_data_random import Region
from tensorflow.keras import backend

# globals for simulated annealing
NUM_ITER = 20000
GEN_ITER = 10
BATCH_SIZE = 50
NUM_BATCH = 100
GP_WEIGHT = 10.0
IGNORE = False # ignores checking for best loss in SA
TEMP = 100.0
print("NUM_ITER", NUM_ITER)
print("BATCH_SIZE", BATCH_SIZE)
print("NUM_BATCH", NUM_BATCH)

# globals for data
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # heuristic to get enough SNPs for simulations
NUM_CLASSES = 2     # "real" vs "simulated"
NUM_CHANNELS = 2    # SNPs and distances
print("NUM_SNPS", NUM_SNPS)
print("L", L)
print("NUM_CLASSES", NUM_CLASSES)
print("NUM_CHANNELS", NUM_CHANNELS)


def main():
    """Parse args and run simulated annealing"""
    opts = util.parse_args()
    print(opts)

    # set up seeds
    if opts.seed != None:
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        tf.random.set_seed(opts.seed)

    generator, discriminator, iterator, parameters = process_opts(opts)

    # grid search
    if opts.grid:
        print("Grid search not supported right now")
        sys.exit()
        # posterior, loss_lst = grid_search(discriminator, samples, simulator, \
        #    iterator, parameters, opts.seed)
    # simulated annealing
    else:
        posterior, loss_lst = simulated_annealing(generator, discriminator,
                                                  iterator, parameters, opts.seed, toy=opts.toy)

    print(posterior)
    print(loss_lst)


def process_opts(opts):

    # parameter defaults
    all_params = util.ParamSet()
    parameters = util.parse_params(opts.params, all_params)  # desired params
    param_names = [p.name for p in parameters]

    # if real data provided
    real = False
    if opts.data_h5 != None:
        # most typical case for real data
        iterator = real_data_random.RealDataRandomIterator(NUM_SNPS,
                                                           opts.data_h5, opts.bed)
        num_samples = iterator.num_samples  # TODO use num_samples below
        real = True

    filter = False  # for filtering singletons

    # parse model and simulator
    if opts.model == 'const':
        sample_sizes = [198]
        discriminator = discriminators.OnePopModel()
        simulator = simulation.simulate_const
        # print("FILTERING SINGLETONS")
        # filter = True

    # exp growth
    elif opts.model == 'exp':
        sample_sizes = [198]
        discriminator = discriminators.OnePopModel()
        simulator = simulation.simulate_exp
        # print("FILTERING SINGLETONS")
        # filter = True

    # isolation-with-migration model (2 populations)
    elif opts.model == 'im':
        sample_sizes = [98, 98]
        discriminator = discriminators.TwoPopModel(sample_sizes[0],
                                                   sample_sizes[1])
        simulator = simulation.simulate_im

    # out-of-Africa model (2 populations)
    elif opts.model == 'ooa2':
        sample_sizes = [98, 98]
        discriminator = discriminators.TwoPopModel(sample_sizes[0],
                                                   sample_sizes[1])
        simulator = simulation.simulate_ooa2

    # CEU/CHB (2 populations)
    elif opts.model == 'post_ooa':
        sample_sizes = [98, 98]
        discriminator = discriminators.TwoPopModel(sample_sizes[0],
                                                   sample_sizes[1])
        simulator = simulation.simulate_postOOA

    # out-of-Africa model (3 populations)
    elif opts.model == 'ooa3':
        sample_sizes = [66, 66, 66]
        # per_pop = int(num_samples/3) # assume equal
        discriminator = discriminators.ThreePopModel(sample_sizes[0],
                                                     sample_sizes[1], sample_sizes[2])
        simulator = simulation.simulate_ooa3

    # no other options
    else:
        sys.exit(opts.model + " is not recognized")

    # generator
    generator = simulation.Generator(simulator, param_names, sample_sizes,
                                     NUM_SNPS, L, opts.seed, mirror_real=real, reco_folder=opts.reco_folder,
                                     filter=filter)

    # "real data" is simulated wiwh fixed params
    if opts.data_h5 == None:
        iterator = simulation.Generator(simulator, param_names, sample_sizes,
                                        NUM_SNPS, L, opts.seed, filter=filter)  # don't need reco_folder

    return generator, discriminator, iterator, parameters

################################################################################
# SIMULATED ANNEALING
################################################################################



def simulated_annealing(generator, discriminator, iterator, parameters, seed,
                        toy=False):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, discriminator, iterator, parameters, seed)

    # NO PRE-TRAINING
    s_current = [param.start() for param in pg_gan.parameters]
    pg_gan.generator.update_params(s_current)

    loss_curr = pg_gan.generator_loss(s_current)
    print("params, loss", s_current, loss_curr)

    posterior = [s_current]
    loss_lst = [loss_curr]
    real_acc_lst = []
    fake_acc_lst = []

    # simulated-annealing iterations
    num_iter = NUM_ITER
    # for toy example
    if toy:
        num_iter = 2

    NA_count = 0

    # main pg-gan loop
    for i in range(num_iter):
        print("\nITER", i)
        print("time", datetime.datetime.now().time())
        T = temperature(i, num_iter)  # reduce width of proposal over time

        # propose 10 updates per param and pick the best one
        s_best = None
        loss_best = float('inf')
        for k in range(len(parameters)):  # trying all params!
            for j in range(GEN_ITER): 

                # can update all the parameters at once, or choose one at a time
                # s_proposal = [parameters[k].proposal(s_current[k], T) for k in\
                #    range(len(parameters))]
                s_proposal = [v for v in s_current]  # copy
                s_proposal[k] = parameters[k].proposal(s_current[k], T)
                loss_proposal = pg_gan.generator_loss(s_proposal)

                print(j, "proposal", s_proposal, loss_proposal)
                if loss_proposal < loss_best:  # minimizing loss
                    loss_best = loss_proposal
                    s_best = s_proposal

        # decide whether to accept or not (reduce accepting bad state later on)

        diff = loss_best - loss_curr

        if loss_best <= loss_curr or IGNORE:  # unsure about this equal here
            p_accept = 1
        else:
            #p_accept = (loss_curr / loss_best) * T
            p_accept = np.exp(-diff/T)
        rand = np.random.rand()
        accept = rand < p_accept

        # if accept, retrain
        if accept:
            print("ACCEPTED")
            s_current = s_best
            generator.update_params(s_current)
            # train only if accept (5 times the generator per parameter)
            #num_batch = 5 * len(parameters) * GEN_ITER
            num_batch = 5
            real_acc, fake_acc, disc_loss, distance, real_loss, fake_loss = pg_gan.train_sa(num_batch, BATCH_SIZE)
            loss_curr = loss_best

        else:
            print("NOT ACCEPTED")
            num_batch = 5
            real_acc, fake_acc, disc_loss, distance, real_loss, fake_loss = pg_gan.train_sa(num_batch, BATCH_SIZE)

        print("T, p_accept, rand, s_current, loss_curr", end=" ")
        print(T, p_accept, rand, s_current, loss_curr)
        posterior.append(s_current)
        loss_lst.append(loss_curr)

    return posterior, loss_lst


def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob."""
    #return 1 - i/num_iter  # start at 1, end at 0
    return TEMP / float(i+1)


################################################################################
# TRAINING
################################################################################

class PG_GAN:

    def __init__(self, generator, discriminator, iterator, parameters, seed):
        """Setup the model and training framework"""
        print("parameters", type(parameters), parameters)

        # set up generator and discriminator
        self.generator = generator
        self.discriminator = discriminator
        self.iterator = iterator  # for training data (real or simulated)
        self.parameters = parameters

        # this checks and prints the model (1 is for the batch size)
        self.discriminator.build_graph((1, iterator.num_samples, NUM_SNPS,
                                        NUM_CHANNELS))
        self.discriminator.summary()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        # STEP6: Use RMSProp Stochastic Gradient Descent
        self.disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    def disc_pretraining(self, num_batches, batch_size):
        """Pre-train so discriminator has a chance to learn before generator"""
        s_best = []
        max_loss = float("-inf")
        k = 0

        # try with several random sets at first
        while k < PRE_TRAIN_TRIALS:
            s_trial = [param.start() for param in self.parameters]
            print("trial", k+1, s_trial)
            self.generator.update_params(s_trial)
            real_acc, fake_acc, disc_loss, distance, real_loss, fake_loss = self.train_sa(num_batches, batch_size)
            if fake_loss > max_loss:
                max_loss = fake_loss
                s_best = s_trial
            k += 1

        # now start!
        self.generator.update_params(s_best)
        return s_best

    def train_sa(self, num_batches, batch_size):
        """Train using fake_values for the simulated data"""

        for epoch in range(num_batches):

            real_regions = self.iterator.real_batch(batch_size, True)
            real_acc, fake_acc, disc_loss, distance, real_loss, fake_loss = self.train_step(real_regions)

            if (epoch+1) % 5 == 0:
                template = 'Epoch {}, Loss: {}, Real Critic Loss: {}, Fake Critic Loss: {}, Distance: {}'
                print(template.format(epoch + 1,
                                      disc_loss,
                                      real_loss,
                                      fake_loss,
									  distance))

        return real_acc/BATCH_SIZE, fake_acc/BATCH_SIZE, disc_loss, distance, real_loss, fake_loss 

    def generator_loss(self, proposed_params):
        """ Generator loss """
        generated_regions = self.generator.simulate_batch(BATCH_SIZE,
                                                          params=proposed_params)
        # not training when we use the discriminator here
        fake_output = self.discriminator(generated_regions, training=False)
		# STEP3: Wasserstein loss is also used here? TODO
        loss = -backend.mean(fake_output)
        # loss = -tf.reduce_mean(fake_output)
        return loss.numpy()

    """
	STEP3 Wasserstein Loss Function
	TODO: fix accuracy to conform with STEP2
	"""

    def wasserstein_loss(self, real_output, fake_output):
        # accuracy
        real_acc = np.sum(real_output <= 0)  # positive logit => pred -1
        fake_acc = np.sum(fake_output > 0)  # negative logit => pred 1


        real_loss = backend.mean(real_output)
        fake_loss = backend.mean(fake_output)
        # real_loss = tf.reduce_mean(real_output)
        # fake_loss = tf.reduce_mean(fake_output)
        distance = 0.5 * (- real_loss + fake_loss)
        total_loss = - real_loss + fake_loss

        return total_loss, real_acc, fake_acc, distance, real_loss, fake_loss

    def discriminator_loss(self, real_output, fake_output):
        """ Discriminator loss """
        # accuracy
        real_acc = np.sum(real_output >= 0)  # positive logit => pred 1
        fake_acc = np.sum(fake_output < 0)  # negative logit => pred 0

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        # add on entropy regularization (small penalty)
        real_entropy = self.cross_entropy(real_output, real_output)
        fake_entropy = self.cross_entropy(fake_output, fake_output)
        entropy = tf.math.scalar_mul(0.001/2, tf.math.add(real_entropy,
                                                          fake_entropy))  # can I just use +,*? TODO
        total_loss -= entropy  # maximize entropy

        return total_loss, real_acc, fake_acc

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """


        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 2], 0.0, 1.0)
        diff = np.subtract(fake_images,real_images)
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_regions):
        """One mini-batch for the discriminator"""

        with tf.GradientTape() as disc_tape:
            # use current params
            generated_regions = self.generator.simulate_batch(BATCH_SIZE)

            real_output = self.discriminator(real_regions, training=True)
            fake_output = self.discriminator(generated_regions, training=True)

            # disc_loss, real_acc, fake_acc = self.discriminator_loss( \
            # real_output, fake_output)
            disc_loss, real_acc, fake_acc, distance, real_loss, fake_loss = self.wasserstein_loss(
                real_output, fake_output)
            gp = self.gradient_penalty(BATCH_SIZE, real_regions, generated_regions)
            disc_loss = disc_loss + gp * GP_WEIGHT

        # gradient descent
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                        self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                self.discriminator.trainable_variables))

        return real_acc, fake_acc, disc_loss, distance, real_loss, fake_loss


if __name__ == "__main__":
    main()
