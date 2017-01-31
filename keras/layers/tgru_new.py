


class TerminalGRU(GRU):

    def __init__(self, output_dim, temperature=1, rnd_seed=None,**kwargs):
        self.output_dim = output_dim
        super(TerminalGRU, self).__init__(**kwargs)
        self.temperature = temperature
        self.rnd_seed = rnd_seed
        self.uses_learning_phase = True

    def build(self, input_shape):
        # what is this Y for? 
        self.Y = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_Y'.format(self.name))
        
        # define weights here, W, U, etc.
        super(TerminalGRU, self).build(input_shape)
        # trainable_weights an attr of Layer
        self.trainable_weights += [self.Y]

    def call(x, mask=None): 
        # logic of the layer
        # do we need masking?

    def get_output_shape_for(input_shape):
        # If output shape is different (which it is, will need to get used to it)
        new_shape = 
        return new_shape


    def step(self, h, states):
        prev_state = states[0][0]
        
        def training_phase(h, states):
            # training phase behavior, using teacher forcing 
            # return correct functions
            if len(states) == 2:
                B_U = states[-1]
            elif len(states) == 1:
                # setting the droput features
                B_U = np.array([1., 1., 1., 1.], dtype='float32')
            elif len(states) > 2:
                raise Exception('States has three elements')

            #!# previous input to train on;
            # see line 271 of terminalgru.py 
            h = #history of some sort? previous state vectors?
            prev_sampled_output = # derived from h[1][:, :self.output_dim]

            # for self.consume_less != 'gpu' only
            x_z = h[;, :self.output_dim]
            x_r = h[:, self.output_dim: 2*self.output_dim]
            x_h = h[:, 2*self.output_dim:]

            z = self.inner_activation(x_z + K.dot(prev_state*B_U[0], self.U_z))
            r = self.inner_activation(x_z + K.dot(prev_state*B_U[1], self.U_r))

            hh = self.activation(x_h + 
                                K.dot( r * prev_state * B_U[2], self.U_h) +
                                K.dot( r * prev_sampled_output * B_U[3], self.Y)) 
            
            output = z * prev_state + (1. - z ) * hh
            final_output = output

            return output, final output 

        def testing_phase(h, states):
            # testing phase behavior, use learned output
            B_U = np.array([1., 1., 1., 1.], dtype = 'float32')
            prev_sampled_output = states[0][1]

            x_z = h[;, :self.output_dim]
            x_r = h[:, self.output_dim: 2*self.output_dim]
            x_h = h[:, 2*self.output_dim:]

            z = self.inner_activation(x_z + K.dot(prev_state*B_U[0], self.U_z))
            r = self.inner_activation(x_z + K.dot(prev_state*B_U[1], self.U_r))

            hh = self.activation(x_h + 
                                K.dot( r * prev_state * B_U[2], self.U_h) +
                                K.dot( r * prev_sampled_output * B_U[3], self.Y)) 
            
            output = z * prev_state + (1. - z ) * hh

            #!# final output should be sampled, see line 301
            sampled_output = 
            return output, final output 

        return K.in_train_phase(training_phase(h, states), testing_phase(h, states))
