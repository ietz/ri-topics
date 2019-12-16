from flask import Flask, request, jsonify

from ri_topics.dtos import Trend, TopicActivity, TopicContent
from ri_topics.topics import TopicModelManager
from ri_topics.trend import find_trends, find_top


class RiTopicsApp(Flask):
    model_manager: TopicModelManager


app = RiTopicsApp(__name__)


@app.route('/<account_name>/trends')
def trends(account_name: str):
    start = request.args.get('start')
    end = request.args.get('end')

    model = app.model_manager.get(account_name)
    trend_df = find_trends(model, start, end).sort_values('score')
    all_trends = [Trend.from_df_tuple(t) for t in trend_df.itertuples()]

    return jsonify({
        'falling': all_trends[:3],
        'rising': all_trends[-3:][::-1],
    })


@app.route('/<account_name>/frequent')
def frequent(account_name: str):
    start = request.args.get('start', default=None)
    end = request.args.get('end', default=None)

    model = app.model_manager.get(account_name)
    top_df = find_top(model, start, end).sort_values('tweet_count', ascending=False)

    return jsonify([TopicActivity.from_df_tuple(t) for t in top_df.itertuples()])


@app.route('/<account_name>/topics/<int:topic_id>')
def topic(account_name: str, topic_id: int):
    model = app.model_manager.get(account_name)

    topic_tuple = model.topic_by_id(topic_id)
    tweet_df = model \
        .tweet_df[model.tweet_df['label'] == topic_id] \
        .sort_values('probability', ascending=False)

    return jsonify(TopicContent.from_df_tuple(topic_tuple, list(tweet_df.index)))
