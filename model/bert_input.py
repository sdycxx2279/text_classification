class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a):
        """Constructs a InputExample.

        Args:
            text_a: string. The untokenized paragraph of the first sequence. For single
                sequence tasks, only this sequence must be specified.
        """
        self.text_a = text_a