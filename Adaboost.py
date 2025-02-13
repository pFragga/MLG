import numpy as np
import math
#commited version to be improved
class Example:

    def __init__(self,x, y, weight):
      self.x = x
      self.y = y
      self.weight = weight

    def update_weight(self,error):
      self.weight = self.weight * error/(1-error)



class Hypothesis:

    def __init__(self,error):
      self.error = error

    def calc_error(self, num_of_texts, word, vector, results, examples):

      error_count = 0
      text = 0
      while text < num_of_texts:
          #calculate errors
          if vector[text][word] != results[text]: #hypothesis
            error_count += examples[text].weight

          text += 1

      #if error>0.5 reverse it
      if error_count > 0.5:
         error_count = 1 - error_count

      self.error = error_count


    def calc_amount_of_say(self):
      if self.error != 0:
        amount_of_say = 1/2 * math.log((1-self.error)/self.error)
        return amount_of_say
      return 100000;


class Adaboost:

  def __init__(self, num_of_loops, vector, results):

    num_of_texts = len(vector)
    num_of_words = np.shape(vector)[1]

    #Initialize Example Array and the weight of each example
    examples = []
    example_weights = []
    i = 0
    while i < num_of_texts:
      example_weights.append(1/num_of_texts)
      examples.append(Example(vector[i], results[i],example_weights[i]))
      i+=1

    #initialize array of hypothesises
    h_array = []
    word = 0
    while word < num_of_words:
      h_array.append(Hypothesis(0))
      word += 1

    #detect errors by text
    self.num_of_loops = num_of_loops
    max_amount_of_say = -1;
    max_amount_of_say_hypothesis = h_array[0]
    for i in range(num_of_loops):
      word = 0
      while word < num_of_words:
        h_array[word].calc_error(num_of_texts, word, vector, results, examples)
        curr_amount_of_say =  h_array[word].calc_amount_of_say()
        if curr_amount_of_say > max_amount_of_say:
          max_amount_of_say = h_array[word].calc_amount_of_say()
          max_amount_of_say_hypothesis = h_array[word]
        word += 1

      #update the weights
      word = 0
      while word < num_of_words:
        i = 0
        h_error = h_array[word].error
        if h_error <= 0.34:
          while i < num_of_texts:
            if vector[i][word] == results[i]:
              examples[i].update_weight(h_error)
            i += 1
        word += 1


      #normalize the weights
      sum_of_weights = 0
      for example in examples:
        sum_of_weights += example.weight

      for example in examples:
        example.weight = example.weight/sum_of_weights












