

#basics
from typing import List, Tuple, DefaultDict, Dict
from itertools import combinations
from itertools import product
from collections import defaultdict
#pytroch
import torch



class Pairer:


    def __get_pairing_fn(self, bidir=False):

        def get_bidir_pair(x, r):
            return list(product(x, repeat=r))

        def get_pairs(x,r):
            return sorted(  # noqa
                            list(combinations(x, r=r)) + [(e, e) for e in x],
                            key=lambda u: (u[0], u[1])
                            )

        if bidir:
            return get_bidir_pair
        else:
            return get_pairs






    def get_pairs(  self,
                    seg_data: dict,
                    batch : dict, 
                    tasks : List[str], 
                    bidir = False,
                ) -> DefaultDict[str, List[List[Tuple[int]]]]:


        get_pairs = self.__get_pairing_fn(bidir=bidir)

        pair_data = defaultdict(lambda: [])

        idx_start = seg_data["seg"]["start"]
        idx_end = seg_data["seg"]["end"]

        #create ids for each pair, based on the index
        idxs = list(get_pairs(range(len(seg_data["seg"]["ids"])), 2))

        #if there are no pairs we skip else we create the permutaions or combinations of pairs
        if idxs:
            p1, p2 = zip(*idxs)  # pairs' seg id
            p1 = torch.tensor(p1, dtype=torch.long, device=batch.device)
            p2 = torch.tensor(p2, dtype=torch.long, device=batch.device)
            # pairs start and end token id.
            start = get_pairs(idx_start, 2)  # type: List[Tuple[int]]
            end = get_pairs(idx_end, 2)  # type: List[Tuple[int]]
            length = len(start)  # number of pairs' segs len  # type: int

        else:
            p1 = torch.empty(0, dtype=torch.long, device=batch.device)
            p2 = torch.empty(0, dtype=torch.long, device=batch.device)
            start = []
            end = []
            lens = 0

        # pair_data["idx"].append(idxs)
        # pair_data["p1"].append(p1)
        # pair_data["p2"].append(p2)
        # pair_data["start"].append(start)
        # pair_data["end"].append(end)
        # pair_data["lengths"].append(length)


        #we also get the true labels for each of the spans
        for task in tasks:
            start_p1, start_p2 = zip(*start)

            # for links we 
            if task == "link":
                pair_data[f"T-{task}"].append(batch["token"][task][i][start])

            else:
                # the true labels for the segments are the samme across all tokens in the segment
                # so we only need to to use start or end index to get the true labels
                pair_data[f"T-{task}_p1"].append(batch["token"][task][i][start_p1])
                pair_data[f"T-{task}_p2"].append(batch["token"][task][i][start_p2])


        pair_data["lengths"] = torch.tensor(
                                            pair_data["lengths"],
                                            dtype=torch.long,
                                            device=batch.device
                                            )

        pair_data["total_pairs"] = sum(pair_data["lengths"])

        return pair_data