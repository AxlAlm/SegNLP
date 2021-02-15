




class ModelOutput:

    def __init__(self, dataset, calc_metrics:bool=True):
        self.dataset = dataset

        self._need_seg_lens = if self.dataset.prediction_level == "token"
        self._seg_lens_added = False
        self._total_loss_added = False

        if "seg" in self.dataset.subtasks:
            seg_task = [t for t in self.dataset.tasks if "seg" in task][0]
            Bs = [i for l,i in self.dataset.encoders[seg_task].label2id.items() if l.lower().startswith("b")]
            Is = [i for l,i in self.dataset.encoders[seg_task].label2id.items() if l.lower().startswith("i")]
            Os = [i for l,i in self.dataset.encoders[seg_task].label2id.items() if l.lower().startswith("o")]
            self.bio = BIO_Decoder(
                                    Bs=Bs,
                                    Is=Is,
                                    Os=Os,
                                    )
        
        
        self.seg = {}
        self.loss = {}
        self.preds = {}
        self.probs = {}
        self.batch = None
        #self.outputs = outputs = {ID:{"preds":{}, "probs":{}, "gold":{}, "text":{}} for ID in id2idx.keys()}


    def __get_subtask_preds(decoded_labels:list, task:str):
        """
        given that some labels are complexed, i.e. 2 plus tasks in one, we can break these apart and 
        get the predictions for each of the task so we can get metric for each of them. 

        for example, if one task is Segmentation+Arugment Component Classification, this task is actually two
        tasks hence we can break the predictions so we cant get scores for each of the tasks.
        """
        subtasks_predictions = {}
        for subtask in task.split("_"):

            # get the positon of the subtask in the joint label
            # e.g. for  labels following this, B_MajorClaim, 
            # BIO is at position 0, and AC is at position 1
            subtask_position = self.dataset.get_subtask_position(task, subtask)
            subtask_labels = [p.split("_")[subtask_position] for p in decoded_labels]

            # remake joint labels to subtask labels
            #subtask_preds = torch.LongTensor(self.dataset.encode_list([p.split("_")[subtask_position] for p in decoded_labels], subtask))
            #subtasks_predictions[subtask] = subtask_preds.view(original_shape)
            subtasks_predictions[subtask] = subtask_labels

        return subtasks_predictions
    

    def __decode(self, preds:list, lengths:list):
        decoded_preds  = []
        for i, sample_preds in enumerate(preds):
            decoded_preds.extend(self.dataset.decode_list(preds[:lenghts[i]], task))
        
        return decoded_preds


    def __get_segment_labels(labels:list, max_nr_segs:int, task:str):

        nr_samples = labels.shape[0]
        seg_labels = torch.zeros(nr_samples, max_nr_segs)
        i = 0
        for i in range(nr_samples):
            floor = 0
            length_type = zip(self.seg["types"][i] , self.seg["lengths"][i])
            # we dont want segments taht are not Argument Components. These ones we filter out.
            sample_seg_lens = [l for t,l in length_type if t is not None] 
            nr_segs = len(sample_seg_lens)
            for j in range(nr_segs):
                    
                seg_labels = labels[i][floor:floor+length]
                most_freq_label = Counter(seg_labels).most_common(1)[0][0]

                if task == "relation":
                    
                    point_to_idx = j + most_freq_label 
                    if point_to_idx > nr_segs or point_to_idx < 0:
                        most_freq_label = nr_segs - j

                seg_labels[i][j] = most_freq_label
                floor += length

        return seg_labels


    def add_loss(self, task:str, data=torch.tensor):

        assert torch.is_tensor(data), f"{task} loss need to be a tensor"
        assert data.requires_grad, f"{task} loss tensor should require grads"

        if task == "total":
            loss["total"] = data
        else:
            loss[task] = data

            if not self._total_loss_added:

                if "total" in loss:
                    loss["total"] = data
                else:
                    loss["total"] += data
    
        self.metrics[f"{task}-loss"] = ensure_numpy(data)


    def add_preds(self, task:str, level:str, data:torch.tensor):

        assert task in set(self.dataset.all_tasks), f"{task} is not a supported task. Supported tasks are: {self.dataset.all_tasks}"
        assert level in set(["token", "ac"]), f"{level} is not a supported level. Only 'token' or 'ac' are supported levels"
        assert torch.is_tensor(data), f"{task} preds need to be a tensor"
        assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"

        # for segmentation we need to get segment lenghts for each sample
        if "seg" in task and not self._seg_lens_added:
            self.seg["lengths"], self.seg["types"]= self.bio.decode(data)
            self._seg_lens_added = True

        # if task is a complex task we get the predictions for each of the subtasks embedded in the 
        # complex task. We will only use these to evaluate
        if "_" in task:
            decoded_labels = __decode(preds=data, lengths=self.lengths)
            subtask_preds = __get_subtask_preds(decoded_labels=decoded_labels, task=task)
            
            for stask, sdata in subtask_preds.items():
                self.add_preds(task=stask, level=level, data=sdata)
        else:
            # If we are prediction on token level we need to convert the labels ot segment labels. For this we need
            # the length of each segment predicted by the model.
            if level == "token":
                if not self._seg_lens_added:
                    raise RuntimeError("When segmentation is a subtasks it needs to be added first to output")

            decoded_labels = __decode(preds=data, lengths=self.lengths)

            seg_labels = self.__get_segment_labels(
                                                        labels=decoded_labels, 
                                                        max_nr_segs=self.dataset.max_nr_segs, 
                                                        task=task
                                                        )
            
            if calc_metrics:
                self.metrics = MetricManager().pred_metrics()


            self.preds[task] = seg_labels


    def add_probs(self, task:str, level:str, data:torch.tensor):

        assert torch.is_tensor(data), f"{task} probs need to be a tensor"
        assert len(data.shape) == 3, f"{task} probs need to be a 3D tensor"


        if calc_metrics:
            self.metrics = MetricManager().pred_metrics()



    def clear(self):
        self.seg = {}
        self.loss = {}
        self.preds = {}
        self.probs = {}
        self.batch = None
        self.outputs


    # outputs = {}

    # #first we decode the complex tasks, and get the predictions for each of the subtasks
    # complex_tasks = [task for task in self.dataset.tasks if "_" in task]
    # for c_task in complex_tasks:
    #     decoded_labels = decode(preds=preds, lengths=lengths)
    #     subtask_preds = get_subtask_preds(decoded_labels=decoded_labels, task=c_task)
    #     output_dict.update(subtask_preds)


    # for task in self.dataset.subtasks:
        
    #     #if we have the decoded labels
    #     if task not in output_dict:
    #         decoded_labels = outputs[task]
        
    #     #if we dont have the decoded labels
    #     else:
    #         decoded_labels = decode(preds=preds, lengths=lengths)
    #         outputs[task] = decoded_labels


    #     get_segment_labels(decoded_labels, segment_lengths, segment_types, max_nr_segs, task)


    # return outputs
    