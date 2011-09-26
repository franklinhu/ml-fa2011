from decision_tree import decision_tree

def decision_tree_learn(examples, attributes, parent_examples):
  # If no examples, use pluraity of parent
  if len(examples) == 0:
    return plurality_value(parent_examples)

  # If all are of the same class, use that class
  common_class = same_class(attribute, examples):
  if common_class:
    return common_class

  # If there are no more attributes left, use plurality
  if len(attributes) == 0:
    return plurality_value(examples)

  A = importance(attributes, examples)
  tree = decision_tree(A)

  raise Exception("UNIMPLEMENTED")

def plurality_value(examples):
  pass

def same_class(attribute, examples):
  pass

def importance(attributes, examples):
  pass

if __name__ == "__main__":
  pass
