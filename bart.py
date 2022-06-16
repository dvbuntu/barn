import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import rpy2.robjects as r
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()

bart = importr('BART')

wbart = bart.wbart

# TODO: set bart params here
def fit_bart(data, ntrees=30, random_state=42, verbose=False, power=2, base=.95, a=0.5, b=1):
    """
    Fits a bart forest to some data and returns the model.

    First builds the model using rpy2, then translates to sklearn format
    """
    # extract data
    XTrain, XValid, XTest, YTrain, YValid, YTest = data

    # convert to R format
    r.r['set.seed'](random_state)
    rx = r.r.matrix(XTrain, nrow=XTrain.shape[0], ncol=XTrain.shape[1])
    ry = r.FloatVector(YTrain)
    rxtest = r.r.matrix(XValid, nrow=XValid.shape[0], ncol=XValid.shape[1])

    # train bart
    if verbose:
        printevery = 100
    else:
        printevery = 100000 # can't see how to silence bart
    rb = wbart(rx,ry,rxtest, nkeeptreedraws=1, ntree=ntrees, power=power, base=base, printevery=printevery, a=a, b=b)

    # lines of trees
    treelines = rb.rx2['treedraws'].rx2['trees'][0].split('\n')

    # need cutpoints as well
    cuts = [np.array(c) for c in rb.rx2['treedraws'].rx2['cutpoints']]

    # translate trees to sklearn
    trees = parse_trees(treelines, cuts)

    # build sklearn model
    model = make_dummy_trees(trees, np.mean(YTrain), XTrain.shape[1])

    # generate predictions
    bart_predictions_train = model.predict(XTrain)
    bart_predictions_valid = model.predict(XValid)
    bart_predictions_test = model.predict(XTest)

    # compute RMSE metrics for predictions
    bart_score_train = np.sqrt(np.mean (np.square(bart_predictions_train-np.squeeze(YTrain) ) ))
    bart_score_valid = np.sqrt(np.mean (np.square(bart_predictions_valid-np.squeeze(YValid) ) ))
    bart_score_test = np.sqrt(np.mean (np.square(bart_predictions_test-np.squeeze(YTest) ) ) )
    if verbose:
        print ("bart score (RMSE) train: ", bart_score_train)
        print ("bart score (RMSE) valid: ", bart_score_valid)
        print ("bart score (RMSE) test: ", bart_score_test)
    model_results = (bart_score_train, bart_score_valid, bart_score_test)

    return model, model_results


def parse_trees(treelines, cuts):
    j1, ntree, j2 = treelines[0].strip().split()
    ntree = int(ntree)
    trees = []
    lineno = 1
    while True:
        try:
            nodes = int(treelines[lineno].strip())
            lineno += 1
        except ValueError:
            break
        rows = dict()
        values = dict()
        old2new = dict()
        new2old = dict()
        old2new[-1] = -1
        new2old[-1] = -1
        maxnode = 0
        for i in range(nodes):
            node, var, cut, th = treelines[lineno].strip().split()
            lineno += 1
            node = int(node)
            if node > maxnode:
                maxnode = node
            var = int(var)
            cut = int(cut)
            thresh = cuts[var][cut]
            val = float(th)
            # need to renumber these to fit expected sklearn pattern
            ## and change to 0-based indexing in python
            row = (2*node, 2*node+1, var, thresh, 1, 1, 1) # dummy values
            rows[node] = row
            values[node] = val
            new2old[i] = node
            old2new[node] = i
        # identify leaf nodes (no children)
        for k,v in rows.items():
            # leaf node
            if 2*k not in rows:
                tmp = rows[k]
                rows[k] = (-1,-1,-2, tmp[3], 0, 1, 1)
        # fix node labels
        for i in range(nodes):
            k = new2old[i]
            tmp = list(rows[k])
            tmp[0] = old2new[rows[k][0]]
            tmp[1] = old2new[rows[k][1]]
            rows[k] = tuple(tmp)
        ## fill in the empty nodes
        ## first node needs to feed into actual start
        row_list = [i for i in range(nodes)]
        values_arr = np.zeros((1,nodes))
        for k in range(nodes):
            j = new2old[k]
            row_list[k] = rows[j]
            values_arr[0,k] = values[j]
        depth = np.floor(np.log2(maxnode))+1
        trees.append((row_list, values_arr, depth))
    return trees

def make_dummy_trees(trees, ymean, n_features):
    ntree = len(trees)

    t_arr = [np.array(rows[0], dtype=[('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'), ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')]) for rows in trees]
    v_arr = [v[1] for v in trees]
    depths = [row[2] for row in trees]

    dummy = RandomForestRegressor(n_estimators=ntree+1)
    dummy.n_outputs_ = 1
    dummy.estimators_ = [DecisionTreeRegressor(max_depth=int(depths[i])+1)
                            for i in range(ntree)]

    for i,t in enumerate(t_arr):
        # dummy fitting, going to throw away
        ## but needs to build a tree big enough
        dummy.estimators_[i].fit(np.random.random((len(t),n_features)), np.arange(len(t)))
        dummy.estimators_[i].tree_.__setstate__({'nodes':t,
                'max_depth':depths[i],
                'node_count':len(t),
                # multiply by ntree+1 b/c rf base divides
                'values': v_arr[i].reshape((len(t_arr[i]),1,1))*(ntree+1)
            })
        dummy.estimators_[i].n_outputs_ = 1

    # add a dummy tree to account for the mean of the training targets
    t = np.array([(1,2,0, -1,-1,-1,-1),( -1,-1,-2, -1,-1,-1,-1),( -1,-1,-2, -1,-1,-1,-1),], dtype=[('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'), ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')])
    dummy.estimators_.append(DecisionTreeRegressor(max_depth=1))
    dummy.estimators_[-1].fit(np.random.random((len(t),n_features)), np.arange(len(t)))
    dummy.estimators_[-1].tree_.__setstate__({'nodes':t,
            'max_depth':2,
            'node_count':3,
            # multiply by ntree+1 b/c rf base divides
            'values': np.array([0, ymean, ymean]).reshape((3,1,1))*(ntree+1)
        })
    dummy.estimators_[-1].n_outputs_ = 1
    dummy.n_features_in = n_features

    return dummy


