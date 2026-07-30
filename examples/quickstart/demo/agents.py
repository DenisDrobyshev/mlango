"""Agents for the demo app."""

from mlango.agents import Agent, BufferMemory, tool


@tool
def classify_review(text: str) -> str:
    """Classify a product review as positive or negative.

    Args:
        text: The review text to classify.
    """
    from demo.models import Sentiment

    try:
        model = Sentiment.load()
    except LookupError:
        return "No trained model yet. Run: python manage.py train demo.Sentiment"
    return str(model.predict(text))


class Helper(Agent):
    """Answers questions about reviews and can call the trained classifier."""

    class Meta:
        system = (
            "You help analyse product reviews. When the user gives you review "
            "text and asks for a verdict, use the classify_review tool rather "
            "than guessing."
        )
        tools = [classify_review]
        memory = BufferMemory(k=20)
        max_steps = 6
