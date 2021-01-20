

from hotam.dashboard import Dashboard
from hotam.database import MongoDB
from hotam.loggers import MongoLogger

if __name__ == "__main__":

	db = MongoDB()
	dashboard = Dashboard(db=db).run_server(
											port=8050,
											debug=True,
											#use_reloader=False,
											)