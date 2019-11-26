from faker import Faker

from ri_topics.models import Tweet, Topic, Trend, OccurrenceCount

fake = Faker()


def generate_tweet() -> Tweet:
    return Tweet(
        username=fake.name(),
        text=fake.text(max_nb_chars=110),
    )


def generate_topic() -> Topic:
    return Topic(
        id=str(fake.random_number()),
        name=None,
        representative=generate_tweet(),
    )


def generate_top_trend() -> Trend:
    score = fake.pyfloat(1, 5, positive=True, min_value=3, max_value=7)
    return generate_trend(score)


def generate_flop_trend() -> Trend:
    score = 1 / fake.pyfloat(1, 5, positive=True, min_value=3, max_value=7)
    return generate_trend(score)


def generate_trend(score: float) -> Trend:
    return Trend(
        id=str(fake.random_number()),
        name=None,
        representative=generate_tweet(),
        trendiness_score=score,
        occurrences=generate_occurence_count(score)
    )


def generate_occurence_count(score: float) -> OccurrenceCount:
    current = fake.randomize_nb_elements(100, min=10, max=1000)
    return OccurrenceCount(
        current=current,
        previous=int(current/score)
    )

