import http

from flask import Flask, request, jsonify
from flask_cors import CORS

from ri_topics.dtos import Topic
from ri_topics.topics import TopicModelManager


class RiTopicsApp(Flask):
    model_manager: TopicModelManager


app = RiTopicsApp(__name__)
CORS(app)


@app.route('/<account_name>/topics/', methods=['GET'])
def frequent(account_name: str):
    model = app.model_manager.get(account_name)
    member_ids_by_label = {
        label: list(ids)
        for label, ids
        in model.tweet_df.reset_index().groupby('label')['status_id']
    }

    return jsonify([Topic.from_df_tuple(t, member_ids_by_label[t.Index]) for t in model.topic_df.itertuples()])


@app.route('/<account_name>/topics/<int:topic_id>/', methods=['PATCH'])
def patch_topic(account_name: str, topic_id: int):
    content = request.get_json()

    model = app.model_manager.get(account_name)

    if 'name' in content:
        model.topic_df.loc[topic_id, 'name'] = content['name']
        app.model_manager.save(model)

    return '', http.HTTPStatus.NO_CONTENT
