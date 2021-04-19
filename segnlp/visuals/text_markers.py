    
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib as mpl
import json
import base64
from lxml import etree, html
from IPython.display import Image
import imgkit
from segnlp.utils import RangeDict

strange_characters = {
                        '``':'"',
                        "''":'"',
                        "’":"'",
                        "‘":"'",
                        "’s":"'s"
                    }



def get_color_hex(cmap_name:str, value=1.0):
    norm = mpl.colors.Normalize(vmin=0.0,vmax=2)
    cmap = cm.get_cmap(cmap_name)
    hex_code = colors.to_hex(cmap(norm(value)))
    return hex_code


def under_overline(hex_code_bot, hex_code_top, token):
    return f'<span style="border-bottom: 3px solid {hex_code_bot}; padding-bottom: 2px; border-top: 3px solid {hex_code_top}; padding-top: 2px;">{token}</span>'


def overline(hex_code, token):
    return f'<span style="border-top: 3px solid {hex_code}; padding-top: 2px;">{token}</span>'


def underline(hex_code, token):
    return f'<span style="border-bottom: 4px solid {hex_code}; padding-bottom: 1px;">{token}</span>'


# def get_span2cmap(span_colors:list, spans:list, span2label:dict):

#     nr_color_ranges = len(span_colors)-1
#     nr_spans = len(spans)-1

#     span2cmap = {}
#     cmap2span = {}

#     ci = 0
#     i = 0
#     duplicate_colors = False
#     while i <= nr_spans:

#         if ci > nr_color_ranges:
#             duplicate_colors = True
#             ci = 0
        
#         span_id = spans[i]
#         color = span_colors[ci]
#         span2cmap[span_id] = color
#         cmap2span[color] = span_id

#         i += 1
#         ci += 1

#     return span2cmap


def get_legend(label2cmap, show_spans:bool, show_scores:bool):

    # legend ='<span> Segmentation: </span><span>Gold <span style="border-top: 4px solid black;"> solid </span><span> |Pred <span style="border-bottom: 4px dashed black;"> dashed </span><br>'
    legend = '<span> Segmentation: </span><span> Ground truth is marked by line above text, predictions marked by lined under text </span><br>'

    label_legend = []
    for label, cmap in label2cmap.items():
        color_hex = get_color_hex(cmap, 1.0)
        label_legend.append(f'<span> || {label}: </span><span style="background-color:{color_hex}; color:{color_hex};">ok</span>')

    legend += "<span> Argument Component Types: </span>" + "".join(label_legend) + "<br>"

    if show_scores:
        certainty_span = [f'<span style="background-color:{get_color_hex("Greys",i)}; color:{get_color_hex("Greys",i)};">ok</span>' for i in np.linspace(0.0, 1.0, num = 10)]
        legend += "<span>Certainty: </span>" + ''.join(certainty_span)

    return legend


# def get_mappings(data:list, key:str):

#     spans = set()
#     label2span = {}
#     span2label = {}

#     for d in data:
#         span_id = d.get(key,{}).get("span_id", None) 
#         label = d.get(key,{}).get("label", None)

#         if span_id is not None:
#             spans.add(span_id)

#         if label is not None and span_id is not None:

#             if label not in label2span and label:
#                 label2span[label] = set()

#             if span_id not in span2label:
#                 span2label[span_id] = label

#             label2span[label].add(span_id)
            
#     spans = sorted(spans)
#     return spans, label2span, span2label


def create_token_idx2span_info(span_lengths, pred_none_spans):

    span_colors = [
                        "#cdad00", "#6eb8c1", "#ff6600", "#2f495e", "#16c2f3", "#c5678c","#ff0000",
                        "#5ac18e", "#420420", "#ffa500", "#0000ff", "#800080", "#f6546a","#daf8e3", "#008000", 
                        "#6897bb", "#ea1853", "#fb2e01", "#696969",  "#b2a6e0", "#877a2b", "#059071", "#170a21", 
                        "#b48961", "#f68683", "#69837c", "#8cd2ff", "#afd78f", "#e28a3b", "#497bca",
                        "#ff4d4d", "#005582","#ffc300","#004444","#cd8500", "#23272a", "#99aab5",
                        "#008744",
                        #"#97ebdb", "#00ff00",
                        #"#ffb6c1", ""#62f184"," "#7fed98", 
        
                        ]

    tokenidx2span_info = RangeDict()
    start = 0
    span_id = 0
    ci = 0
    for i,span_length in enumerate(span_lengths):
        span = (start, start+span_length)

        if pred_none_spans[i]:

            if ci > len(span_colors)-1:
                ci = 0

            tokenidx2span_info[span] = {"id":span_id, "hexcode":span_colors[ci]}
            ci += 1

        else:
            tokenidx2span_info[span] = {"id":None, "hexcode":None}

        span_id += 1
        start += span_length

    return tokenidx2span_info


def highlight_text(
                    tokens,
                    labels,
                    pred_labels,
                    pred_span_lengths,
                    pred_none_spans,
                    gold_labels=None,
                    gold_span_lengths=None,
                    gold_none_spans=None,
                    save_path:str="/tmp/highlight_text.png", 
                    return_html:bool=False, 
                    show_spans:bool=True, 
                    show_scores:bool=True, 
                    show_legend:bool=True,
                    font:str="Verdana", 
                    width:int=1000, 
                    height:int=800,
                    prefix=""
                    ):

    assert len(labels) < 8, "Currently only supporting 8 labels"
    show_pred = True
    show_gold = False
    style_elems = [ 
                    "span { line-height: 30px; font-size:small;}",
                   ]

    puncts = set([".", "?", "!"])
    puncts_plus = set([","]) | puncts

    label_colors =  [
                     'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','Greys','YlOrBr', 'RdPu',
                    #  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    #  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
                     ]
    label2cmap = {l:label_colors[i] for i,l in enumerate(labels)}
    
    
    pred_token_idx2span_info = create_token_idx2span_info(pred_span_lengths, pred_none_spans)

    if gold_span_lengths is not None:
        gold_token_idx2span_info = create_token_idx2span_info(gold_span_lengths, gold_none_spans)


    if show_legend:
        legend = "<hr>" + get_legend(
                                label2cmap=label2cmap,
                                show_spans=show_spans, 
                                show_scores=show_scores
                                )
    else:
        legend = ""

    gold_spans = []
    pred_spans = []
    token_stack = []

    make_upper = True
    for i,token in enumerate(tokens):

        pred_span_id = pred_token_idx2span_info[i]["id"]
        pred_span_hexcode = pred_token_idx2span_info[i]["hexcode"]

        pred_label = None
        if show_scores:
            pred_label = None if pred_labels[i] == "None" else pred_labels[i]

        score = 1.0

        gold_span_id = None
        if show_gold:
            gold_span_id = gold_token_idx2span_info[i]["id"]
            gold_span_hexcode = gold_token_idx2span_info[i]["hexcode"]
            gold_label = None if gold_labels[i] == "None" else gold_labels[i]

        if pred_label not in labels:
            pred_label = None

        token = strange_characters.get(token,token)
        
        ## Fixing Capitalization
        if make_upper:
            token = token.capitalize()
            make_upper = False
        
        if token in puncts:
            make_upper = True
        
        try:
            next_token = tokens[i+1]
        except IndexError as e:
            next_token = " "

        ## Fixing spaces
        if next_token not in puncts_plus:
            token = f"{token} "

        if show_scores and show_pred:

            if pred_label is not None:
                color_hex = get_color_hex(label2cmap[pred_label], score)
                token =  f'<span style="background-color:{color_hex};">{token}</span>'

        if show_spans:
  
            if pred_span_id is not None and gold_span_id is not None and show_pred and show_gold:
                token = under_overline(pred_span_hexcode, gold_span_hexcode, token)

            elif pred_span_id is not None and show_pred:
                token = underline(pred_span_hexcode, token)

            elif gold_span_id is not None and show_gold:
                token = overline(gold_span_hexcode, token)


        if ">" not in token:
            token = f'<span>{token}</span>'

        token_stack.append(token)


    #span {{ font-family:"verdana", monospace; font-size: 20px; }} 
    html_string =  f"""<html>
                            <head>
                                <style>
                                {' '.join(style_elems)}
                                </style>
                            </head>
                            <body style="font-family:{font}; font-size:20px;">
                            <br>
                            {prefix}
                            <br>
                            <br>
                            {''.join(token_stack)}
                            <br>
                            {legend}
                            </body>
                            </html>
                    """

    if save_path:
        imgkit.from_string(html_string, save_path, options={'quiet':'', "width": width, "height":height})
    #Image(save_path)

    if return_html:
        document_root = html.fromstring(html_string)
        return etree.tostring(document_root, encoding='unicode', pretty_print=True)