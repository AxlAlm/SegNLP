
#dash
import dash

#hotam
from hotam.dashboard.views.exp_view import ExpView


class DummyDash:

    def __init__(self, db):
        self.app = dash.Dash("Dummy Dash Board")
        self.app.layout = ExpView(self.app, db).layout
    
    def run_server(self, *args, **kwargs):
        self.app.run_server(*args, **kwargs)


