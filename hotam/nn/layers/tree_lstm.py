


class TreeLSTMCell(nn.Module):

    def __init__(self, xemb_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(xemb_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))

        self.W_f = nn.Linear(xemb_size, h_size, bias=False)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)
        self.b_f = nn.Parameter(th.zeros(1, h_size))


    def __lstm_step(self, node_inputs, mask):

        #average node inputs?
        node_inputs = masked_sum(node_inputs, mask)

        #do the lstm thingy here

        return node_outputs 


    def forward(self, features, links, subgraphs=None):

        node_masks = # nodes in the form of maskes over the output, ORDRED by level in the graph
        node_types = # ???
        feature_space = # features for all tokens

        # if subgraphs is passed, shape undecided
        # we expand the batch to cover all subgraphs
        if subgraphs:
            pass
        
        nr_timesteps = max_nodes
        for i in range(nr_timesteps):
            
            # for each node position we get the mask over the feature space
            # then we get the features vectors that are input to that node
            # e.g. for a leaf node input would be word embedding, for a node with 3 children
            # it would be the children vectors

            node_mask = node_masks[:,i]
            node_inputs = feature_space[node_mask]

            node_lengths = torch.sum(node_mask, dim=-1)
            s = torch.split(nf, tuple(node_lengths))
            p = pad_sequence(s, batch_first=True)

            mask = ""
            node_outputs = self.__lstm_step(node_inputs, mask)
            feature_space[:i] = node_outputs

        return feature_space
    




######### we add this to hotam.features, so all the info we need is allready preprocessed
class DepGraph:

    def __parse_graph(self, nodes:set, tree:tuple, nr_nodes:int, level=0):
        idx = []
        levels = []
        masks = [np.zeros(nr_nodes).tolist()]
        masks[0][tree[0]] = 1
        
        if level == 0:
            levels.append(-1)
            idx.append(tree[0])

        for i,node in enumerate(nodes):
            
            if node[1] == tree[0]:
                levels.append(level)
                idx.append(node[0])
                
                filtered = nodes.copy()
                filtered.remove(node)
                
                c_idx , c_levels, c_masks = parse_graph(
                                                            tree=node, 
                                                            nodes=filtered,
                                                            level=level+1,
                                                            nr_nodes=nr_nodes
                                                            )
                
                masks[0][node[0]] = 1
                
                idx.extend(c_idx)
                levels.extend(c_levels)
                masks.extend(c_masks)

        return idx, levels, masks


    def __graph_info(self, links, root_idx):
        
        nodes = list(enumerate(links))
        nr_nodes = len(nodes)
        root = nodes.pop(root_idx)
        idx, levels, masks  = self.__parse_graph(
                                            nodes=nodes, 
                                            tree=root,
                                            nr_nodes=nr_nodes
                                            )

        idx = np.array(idx)
        levels = np.array(levels)
        masks = np.array(masks)
        
        sort_idx = np.argsort(levels)[::-1]
        
        idx = idx[sort_idx]
        levels = levels[sort_idx]
        masks = masks[sort_idx]
        sort_idx
        
        return idx, levels, masks, sort_idx

    
    def extract(self, df):
        sentences  = df.groupby("sentence_id")   

        last_sent_end = None
        links = []
        root_idx = None
        for sent_id, sent_df in enumerate(sentences):
            
            sent_deprels = sent_df["deprel"].to_numpy()
            sent_depheads = sent_df["dephead"].to_numpy()

            sent_root_id = self.encode_list(["root"], "deprel")[0]
            sent_root_idx = int(np.where(sent_df["deprel"].to_numpy() == sent_root_id)[0])
            

            if last_sent_end is not None:
                sent_depheads[sent_root_idx] = last_sent_end
                last_sent_end = sent_df.shape[0]
            else:
                root_idx = sent_root_idx
                last_sent_end = sent_root_idx
            
            links.extend(sent_depheads)


        out = self.__graph_info(links, root_idx)
        return out