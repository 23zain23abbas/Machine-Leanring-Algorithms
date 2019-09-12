import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        predictions = []
        for i in range(len(features)):
            pred = self.root_node.predict(features[i])
            predictions.append(pred)
        return predictions


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        if self.splittable == False:
            return

        if (len(self.features)== 0):
            self.splittable == False
            return

        num_cls = self.num_cls	# Number of Unique  Labels
        allLabels = np.unique(self.labels)	# List of Unique Labels
        labels = self.labels # All Labels
        features = np.array(self.features)
        n_features = features.shape[1]		# Number of Distinct Attributes
        d_features = features.shape[0]		# Number of Data in DataSets

        if n_features == 1:
            self.splittable == False
            return

        featU = [] # Unique Feature Attributes for each Attribute
        featC = [] # Number of Unique Feature Attrbutes
        for i in range(n_features):
            featU.append(np.unique(features[:,i]))
            featC.append(len(featU[i]))

            
        bestIG = -1
        bestAttributeIndex = -1
        for j in range(n_features):
            #DONT LEAVE UNTIL YOU GET IG
            f = features[:,j]
            allAtt = featU[j]
            allCount = featC[j]	 # Number of Feature Unique Values
            labArray = []
            for cls in range(num_cls):
                arr = [0] * allCount
                labArray.append(arr)

            for k in range(len(f)):
                att = f[k]
                lab = labels[k]
                for m in range(allCount):
                    if att == allAtt[m]:
                        break

                loc = m
                for l in range(len(allLabels)):
                    if lab == allLabels[l]:
                        labArray[l][m] += 1

            #Rearranging Arrays
            branches = np.zeros((allCount, num_cls))
            for a in range(len(branches)):
                for b in range(num_cls):
                    branches[a][b] = labArray[b][a]

            sTotal = branches.sum() 
            S = 0
            
            for i in branches:
                si = sum(i)
#                 si = branches[:,i].sum()
                if si == 0:
                    si = 0
                    sl = 0
                else:
                    si = si/sTotal
                    sl = np.log2(si)

                S = S + (-si*sl)

            # Getting IG
            branches = branches.tolist()
            currentIG = Util.Information_Gain(S, branches)
            if currentIG > bestIG:
                bestIG = currentIG
                bestAttributeIndex = j
            elif currentIG == bestIG:
                Unique_Values_Prev =  featC[bestAttributeIndex]
                Unique_Values_Curr = featC[j]
                if Unique_Values_Curr > Unique_Values_Prev:
                    bestIG = currentIG
                    bestAttributeIndex = j
                elif Unique_Values_Curr == Unique_Values_Prev:
                    pass

        if bestIG == -1:
            self.splittable == False
            return


        # Split according to Unique Values
        uValuesCount = featC[bestAttributeIndex]
        uValues = featU[bestAttributeIndex]
        self.dim_split = bestAttributeIndex  # the index of the feature to be split
        self.feature_uniq_split = uValues  # the possible unique values of the feature to be split

        features = np.array(self.features)
        dim_split = self.dim_split
        for i in range(uValuesCount):
            reducedFeatures = []
            reducedLabels = []

            for j in range(len(features)):
                if features[j][dim_split] == uValues[i]:
                    reducedLabels.append(labels[j])
                    #print("\n", "Feats:", features[j],"-->", uValues[i])
                    reducedFeatures.append(features[j])

            self.children.append(TreeNode(reducedFeatures, reducedLabels, num_cls))     
            #print("")


        for i in range(len(self.children)):
            if self.children[i].splittable == True:
                self.children[i].split()

        return

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        
        if self.splittable == False:
            return self.cls_max
        if len(self.features) == 0:
            return self.cls_max
        if self.dim_split == None:
            return self.cls_max
        
        
        featVal = feature[self.dim_split]
        if featVal not in list(self.feature_uniq_split):
            # Error check
            return self.cls_max
        
        featIndex = list(self.feature_uniq_split).index(featVal)
        if featIndex >= len(self.children):
            return self.cls_max
        
        featChild = self.children[featIndex]
        return featChild.predict(feature)
