"""What counts as good enough on this benchmark."""

from mlango.evals import Eval, exact_match

from news.datasets import Posts
from news.models import Topic


class TopicAccuracy(Eval):
    """Does the classifier put a post in the group it came from?

    The threshold is a floor, not a target. A published linear baseline on these
    four groups with the headers stripped lands in the high eighties, so 0.80
    fails loudly on a real regression while leaving room for the run-to-run
    variation any split introduces.
    """

    class Meta:
        dataset = Posts
        target = Topic
        input_field = "text"
        expected_field = "group"
        case_id_field = "id"
        scorers = {"correct": exact_match}
        threshold = 0.80
        max_cases = 500
