

import dash
from data.mongo_data import MongoData

#init the mongo db
db = MongoData()

# Initialise the app
app = dash.Dash(__name__)
