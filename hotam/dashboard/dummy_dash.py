#basic
from multiprocessing import Process


#dash
import dash
import dash_html_components as html
import dash_core_components as dcc

#hotam
from hotam.dashboard.views.live_view import LiveView


class DummyDash:

    def __init__(self, db):
        self.app = dash.Dash("Dummy Dash Board")    
        self.app.layout = html.Div([  
                                    ExpView(self.app, db).layout,
                                    dcc.Interval(
                                                    id='interval-component',
                                                    interval=1*5000, # in milliseconds
                                                    n_intervals=0,
                                                    max_intervals=-1,
                                                )
                                     ])


    def run_server(self, *args, **kwargs):
        dashboard = Process(
                                target=self.app.run_server, 
                                args=args,
                                kwargs=kwargs,
                            )
        dashboard.start()






        #self.app.run_server(*args, **kwargs)


    # def __init__(self, exp_logger=None, db=None):

    #     if exp_logger == None and db == None:
    #         self.db = MongoDB()
    #         self.exp_logger = MongoLogger(db=self.db)
    #     else:
    #         raise NotImplementedError()

    #     assert self.exp_logger.db == self.db, "Logger.db and db is not the same"
    #     self.__dashboard_off = True

    
    #            # if self.__dashboard_off and not debug_mode:
    #         #     #self.start_dashboard(debug_mode=debug_mode)
    #         #     dashboard = Process(
    #         #                             target=self.start_dashboard, 
    #         #                             args=(True, )
    #         #                         )
    #         #     dashboard.start()
    #         #     self.__dashboard_off = False
    # def start_dashboard(self, debug_mode):
    #     dashboard = DummyDash(db=self.db)
    #     webbrowser.open("http://127.0.0.1:8050/")
    #     dashboard.run_server(
    #                             port=8050,
    #                             debug=debug_mode
    #                             )
