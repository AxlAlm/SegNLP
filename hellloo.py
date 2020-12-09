
# from hotam import ExperimentManager
# from hotam.datasets import PE
# from hotam.nn.models import LSTM_CRF
# from hotam.database import DummyDB
# from hotam.features import Embeddings



# if __name__ == "__main__":
# 	pe = PE()
# 	pe.setup(
# 			tasks=["seg"],
# 			multitasks=[], 
# 			sample_level="sentence",
# 			prediction_level="token",
# 			encodings=[],
# 			features=[
# 						Embeddings("glove")
# 						],
# 			remove_duplicates=False,
# 			)
	
# 	hp =  {
# 			"optimizer": "sgd",
# 			"lr": 0.001,
# 			"hidden_dim": 256,
# 			"num_layers": 2,
# 			"bidir": True,
# 			"fine_tune_embs": False,
# 			"batch_size": 10,
# 			"max_epochs":3,
# 			}

# 	ta = {
# 			"logger":None,
# 			"checkpoint_callback":False,
# 			"early_stop_callback":False,
# 			"progress_bar_refresh_rate":1,
# 			"check_val_every_n_epoch":1,
# 			"gpus":None,
# 			#"gpus": [1],
# 			"num_sanity_val_steps":1,  
# 			"overfit_batches":0.01
# 			}


# 	M = ExperimentManager()
# 	M.run( 
# 			project="seg",
# 			dataset=pe,
# 			model=LSTM_CRF,
# 			hyperparamaters=hp,
# 			trainer_args=ta,
# 			monitor_metric="val-seg-f1",
# 			progress_bar_metrics=["val-seg-f1"],
# 			debug_mode=True,
# 			)

# # import dash
# # import dash_html_components as html
# # from dash.dependencies import Output, Input, State
# # import dash_core_components as dcc


# # app = dash.Dash(__name__)


# # class TEST:
    
# #     def __init__(self):
# #         self.xx = "this"
# #         self.v = html.Div(  
# #                             id="test-id",
# #                             children=[
# #                                         html.Div(id="text", children=[html.H2("START")]),
# #                                         dcc.Dropdown(
# #                                                     id=f'text-dropdown',
# #                                                     options=[{"label":v, "value":v} for v in ["1", "3", "okok"]],
# #                                                     value="START",
# #                                                     ),
# #                                         ]
# #                         )
    
# #         app.callback(
# #                     Output('text', 'children'),
# #                     [Input('text-dropdown', 'value')]
# #                     )(self.test_callback)

                    

# #     def test_callback(self, v):
# #         print(v)
# #         return html.H2(self.xx + v)


# # t = TEST()

# # app.layout =  t.v

# # if __name__ == '__main__':
# #     app.run_server(
# #                     debug=True,
# #                     )



from pymongo import MongoClient
client = MongoClient()
db = client['example-database']
collection = db['example-collection']

