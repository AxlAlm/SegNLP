


# import plotly.express as px
# countries_to_hide = ["Australia"]
# df = px.data.gapminder().query("continent=='Oceania'")
# fig = px.line(df, x="year", y="lifeExp", color='country')

# print(fig.legends)

# fig.for_each_trace(lambda trace: trace.update(visible="legendonly") 
#                    if trace.name in countries_to_hide else ())


# k = {}
# fig.for_each_trace(lambda trace: k.update({trace.name:trace.visible}))

# print(k)

# fig.show()



import pymongo
import os

#client = pymongo.MongoClient()
client = pymongo.MongoClient(os.environ['MONGO_KEY'])

for db in client.list_databases():
    print(db)

my_db = client["xalmax"]
for c in my_db.collection_names():
    print(c)
    my_db.drop_collection(c)

# db = client['dummy_db']
# for e in list(db["experiments"].find()):
#     print(e["experiment_id"], e["status"])
#client.drop_database('dummy_db')


#db = client["dummy_db"]

#print(db.list_collection_names())
# from hotam.database import MongoDB
# from hotam.dashboard import DummyDash

# db = MongoDB()

# f = {"experiment_id":'LSTM_CRF_8c0d48ac-8'}
# #f = {"experiment_id":'LSTM_CRF_0571678c-f'}

# print(db.get_experiments(f))
# print(db.get_last_epoch(f))
# print(db.get_scores(f))


# print(db.outputs.find_one(f))


# d = DummyDash(db=db)
# d.run_server(
#             port=8050,
#             debug=True
#             )
