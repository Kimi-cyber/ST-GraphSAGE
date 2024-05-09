import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import networkx as nx

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from tensorflow.keras import layers
from tensorflow.keras import Model

#%%
df = pd.read_csv('STGCNdata.csv') #  Due to confidentiality agreements with the data provider company, we cannot share the original dataset. 
                                    # However, I am currently preparing a synthetic dataset that you can use to test the model. 
                                    # This dataset will soon be made available on GitHub.
                                    # If you have any questions about the model, feel free to contact us.
                                    # email address: juliana2@ualberta.ca; ziming4@ualberta.ca

df = df.drop('CDWater_BBLPerDAY',axis=1)
feature_no = len(list(pdf))
# Extract
data_cols = list(list(pdf)[i] for i in list(range(1,feature_no)))
print(data_cols)
data = df[data_cols].astype(float).to_numpy()
# qg index
qg_idx = data_cols.index('CDGas_MCFPerDAY')
shut_idx = data_cols.index('ShutIns')
print(qg_idx,shut_idx)

# Data Normalozation
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

max_qg = np.max(data[:,qg_idx])
min_qg = np.min(data[:,qg_idx])
print(max_qg)
print(min_qg)
print(norm_data.shape)

t_steps = 48 #reshaped time steps
n_wells = int(len(df)/t_steps)
plot_num = 172
data_total = norm_data.reshape(n_wells,t_steps,data.shape[1])
data_total_raw = data.reshape(n_wells,t_steps,data.shape[1])

#%%
# extract location info
longi_idx = data_cols.index('Longitude')
lati_idx = data_cols.index('Latitude')
print(longi_idx)
print(lati_idx)
loc_info = data_total_raw[:,0,[longi_idx,lati_idx]]
print(loc_info.shape)
plt.plot(loc_info[:,0], loc_info[:,1],'o',markersize=1)

# Cluster the wells by pads
clustering = DBSCAN(eps=0.001,min_samples=1).fit(loc_info)
plt.figure(figsize=(12,8))
cluster_matrix = clustering.labels_
unique_clusters = np.unique(cluster_matrix)
cluster_amts = []
cluster_indices_store = []
for i in unique_clusters:
    cluster_indices = np.where(cluster_matrix  == i)[0]
    cluster_indices_store.append(cluster_indices)
    cluster_coordinates = loc_info[cluster_indices]
    cluster_amt = len(cluster_coordinates)
    cluster_amts.append(cluster_amt)
    mean_x = np.mean(cluster_coordinates[:, 0])
    mean_y = np.mean(cluster_coordinates[:, 1])
    plt.plot(mean_x,mean_y,
                label=f'Cluster {i}',alpha=0.8, marker='o',markersize=10)
    plt.annotate(f'{cluster_amt}', [mean_x,mean_y], ha='center', va='center',fontsize=10,c='w')
# plt.title('Clustered Wells by Pads')
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)

#%% Form the input and the output sequence
split_time = 12
feature_list = list(range(len(list(df))-1))
exampt_list = [i for i in feature_list if i not in [qg_idx]]

# Training data
tr_por = 2240
data_train = data_total[:tr_por,:,:]
x = data_train
print(x.shape)

encoder_inputs = x[:,:split_time,:]
decoder_inputs = x[:,split_time:,exampt_list]
labels = x[:,split_time:,[qg_idx]]

print('Encoder Input shape == {}'.format(encoder_inputs.shape))
print('Decoder Input shape == {}'.format(decoder_inputs.shape))
print('Label shape == {}'.format(labels.shape))

#%% Define Graph Data
def create_adjacency_matrix(sample_points, threshold):
    # Compute the distance matrix using Euclidean distance
    dist_matrix = cdist(sample_points, sample_points)

    # Create the adjacency matrix based on the distance scores and threshold
    adjacency_matrix = np.where(dist_matrix <= threshold, 1, 0) 
#     adjacency_matrix = adjacency_matrix + np.identity(sample_points.shape[0]) # make diagonal equal to 1

    return adjacency_matrix,dist_matrix

min_len = 3
cluster_indices_store2 = []
for i in range(len(cluster_indices_store)):
    cluster_indices = cluster_indices_store[i]
    if len(cluster_indices) >= min_len:
        cluster_indices_store2.append(cluster_indices)
        
with open("cluster_indices_store2.pickle", "wb") as file:
    pickle.dump(cluster_indices_store2, file)
    file.close()
    
threshold = 0.001
tr_split = 280
te_split = len(cluster_indices_store2)-tr_split
adj_tr_store = []
x_tr_store = []
y_tr_store = []
dx_tr_store = []
for i in range(tr_split):
    adj_matrix, _ = create_adjacency_matrix(loc_info[cluster_indices_store2[i]], threshold)
    adj_tr_store.append(adj_matrix)
    x_tr = encoder_inputs[cluster_indices_store2[i],:,:]
    x_tr_store.append(x_tr)
    y_tr = labels[cluster_indices_store2[i],:,:]
    y_tr_store.append(y_tr)
    dx_tr = decoder_inputs[cluster_indices_store2[i],:,:]
    dx_tr_store.append(dx_tr)

adj_te_store = []
x_te_store = []
y_te_store = []
dx_te_store = []
for i in range(tr_split,len(cluster_indices_store2)):
    adj_matrix, _ = create_adjacency_matrix(loc_info[cluster_indices_store2[i]], threshold)
    adj_te_store.append(adj_matrix)
    x_te = encoder_inputs[cluster_indices_store2[i],:,:]
    x_te_store.append(x_te)
    y_te = labels[cluster_indices_store2[i],:,:]
    y_te_store.append(y_te)
    dx_te = decoder_inputs[cluster_indices_store2[i],:,:]
    dx_te_store.append(dx_te)
    
#%% Form graph
dj_tr = nx.disjoint_union_all(
    [nx.from_numpy_array(adj_matrix) for adj_matrix in adj_tr_store])
a_tr = np.array(nx.to_numpy_matrix(dj_tr))

dj_te = nx.disjoint_union_all(
    [nx.from_numpy_array(adj_matrix) for adj_matrix in adj_te_store])
a_te = np.array(nx.to_numpy_matrix(dj_te))

x_tr = np.concatenate(x_tr_store, axis=0)
x_te = np.concatenate(x_te_store, axis=0)
y_tr = np.concatenate(y_tr_store, axis=0)
y_te = np.concatenate(y_te_store, axis=0)
dx_tr = np.concatenate(dx_tr_store, axis=0)
dx_te = np.concatenate(dx_te_store, axis=0)

print('encoder x:',x_tr.shape,'a:',a_tr.shape,'y:',y_tr.shape,'decoder x:',dx_tr.shape)
print('encoder x:',x_te.shape,'a:',a_te.shape,'y:',y_te.shape,'decoder x:',dx_te.shape)

#%% Build GraphSAGE Network
class GraphSAGE(layers.Layer):
    def __init__(
        self,
        out_feat,
        encoder_seq_len,
        **kwargs):
        
        super().__init__(**kwargs)
        self.out_feat = out_feat
        self.seq_len = encoder_seq_len
        self.activation = layers.Activation('relu')
        self.dense_embedding = layers.Dense(self.out_feat, activation=None)
        self.dense_pool = layers.Dense(self.out_feat, activation='relu')
        self.dense_sage = layers.Dense(self.out_feat, activation='relu')

    def aggregate(self, neighbour_representations: tf.Tensor, s0, num_nodes):
        aggregation_func = tf.math.segment_max
        aggregation_info = aggregation_func(neighbour_representations, s0)
        return aggregation_info

    def compute_nodes_representation(self, features: tf.Tensor):
        h_v = self.dense_embedding(features)
        return h_v

    def compute_aggregated_messages(self, features: tf.Tensor, s1,s0):
        neighbour_representations = tf.gather(features, s1)
        weighted_message = self.dense_pool(neighbour_representations)
        h_v = self.aggregate(weighted_message, s0, features.shape[0])
        return h_v
        
    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        h_concat = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        h = self.dense_sage(h_concat)
        return h

    def call(self, inputs):
        features, adj_matrix = inputs
        
        # Message
        nodes_representation = self.compute_nodes_representation(features)
        
        # Aggregation
        indices = tf.where(adj_matrix != 0)
        s1 = indices[:,1] # Nodes receiving the message
        s0 = indices[:,0] # Nodes sending the message (ie neighbors)
        aggregated_messages = self.compute_aggregated_messages(features,s1,s0)
        
        # Update
        update_messages = self.update(nodes_representation, aggregated_messages)
        
        # L2 Nrom
        norm_messages = tf.math.l2_normalize(update_messages, axis=1)
        
        return norm_messages
    
    # allow time-distributed layer
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        output_shape = (batch_size, self.seq_len, self.out_feat)  # Define the appropriate shape for the output tensor
        return output_shape
    
class STSage(layers.Layer):

    def __init__(self,gcn_unit,encoder_seq_len,**kwargs):
        super().__init__(**kwargs)

        # graph conv layer
        self.gru1 = layers.GRU(64,return_sequences=True)
        self.graph_conv = layers.TimeDistributed(GraphSAGE(gcn_unit,encoder_seq_len))
        self.gru2 = layers.GRU(64,return_sequences=True)
        self.drop = layers.Dropout(0.1)
        self.gru3 = layers.GRU(64,return_sequences=False)

    def call(self, inputs):
        x_in, a_in = inputs

        h = self.gru1(x_in)
        gcn_out = self.graph_conv([h,a_in]) 
        h = self.gru2(gcn_out) 
        h = self.drop(h)
        h_out = self.gru3(h)
        
        return h_out

class SAGENet(Model):
    def __init__(self, gcn_unit, encoder_seq_len, out_feat,**kwargs):
        super().__init__(**kwargs)
        
        self.stgcn = STSage(gcn_unit,encoder_seq_len)
        self.gru1 = layers.GRU(64,return_sequences=True)
        self.drop1 = layers.Dropout(0.1)
        self.gru2 = layers.GRU(64,return_sequences=True)
        self.drop2 = layers.Dropout(0.1)
        self.dense1 = layers.Dense(128,activation="tanh")
        self.dense2 = layers.Dense(out_feat,activation="tanh")

    def call(self, inputs):
        encoder_x_in, decoder_x_in, a_in = inputs
        
        encoder_graph = self.stgcn([encoder_x_in,a_in])
        
        h = self.gru1(decoder_x_in, initial_state=encoder_graph) 
        h = self.drop1(h) 
        h = self.gru2(h, initial_state=encoder_graph) 
        h = self.drop2(h)
        h = self.dense1(h)  
        h_out = self.dense2(h)  
        return h_out
    
gcn_unit = 64
encoder_seq_len = 12
outseq_feat = 1

SAGEmodel = SAGENet(gcn_unit, encoder_seq_len, outseq_feat)
tf.config.run_functions_eagerly(False)
SAGEmodel.compile(loss="mse", optimizer='adam')

adj_matrix_train = np.tile(a_tr, (x_tr.shape[1], 1, 1))
adj_matrix_train = np.transpose(adj_matrix_train,(1,0,2))
print(adj_matrix_train.shape)

adj_matrix_test = np.tile(a_te, (x_te.shape[1], 1, 1))
adj_matrix_test = np.transpose(adj_matrix_test,(1,0,2))
print(adj_matrix_test.shape)

#%% Train the model
epochs = 500
SAGEmodel.fit(
    [x_tr,dx_tr,adj_matrix_train], y_tr,
#     validation_data=([x_va,adj_val_padded], y_va),
    batch_size= x_tr.shape[0],
    epochs=epochs)

filename="GraphEncoder_mean_weight1"
filepath = filename+".hdf5"
SAGEmodel.save_weights(filepath)

max_features = adj_matrix_train.shape[-1]  # maximum number of features in training adj_matrix
adj_te_padded = np.pad(adj_matrix_test, [(0,0), (0,0), (0, max_features - adj_matrix_test.shape[-1])], mode='constant')
print(adj_te_padded.shape)
y_pred = SAGEmodel.predict([x_te,dx_te,adj_te_padded], batch_size= x_te.shape[0])
print(y_pred.shape)

#%% Prediction
filename="GraphEncoder_weight3"
filepath = filename+".hdf5"
SAGEmodel.load_weights(filepath)
y_pred = SAGEmodel.predict([x_te,dx_te,adj_te_padded])

def local_RMSE(testY_org,testPredict):
    # Local RMSE
    scaler = MinMaxScaler()
    scaler.fit(testY_org.reshape(-1,1))
    norm_y_label = scaler.transform(testY_org.reshape(-1,1))
    norm_y_pred = scaler.transform(testPredict.reshape(-1,1))
    testScore = mean_squared_error(norm_y_label, norm_y_pred,squared=False)
    return testScore

row=9
col=4
n=0
x_qg = np.concatenate((x_te[:,:,[qg_idx]],y_te),axis=1)
prod_period=12
testScore_all = []
maxx = max_qg
minn = min_qg
fig, ax = plt.subplots(row, col, figsize=(15,21))
for i in range(row):
    for j in range(col):
        plotY_pred = y_pred[n,:,:]*(maxx - minn) + minn
        y_label_val = y_te[n,:,:]*(maxx - minn) + minn
        plot_label = x_qg[n,:,:]*(maxx - minn) + minn
        testScore = local_RMSE(plotY_pred,y_label_val)
        testScore_all.append(testScore)
        
        ax[i,j].plot(list(range(t_steps)),plot_label,'k-o',alpha=0.1,label= 'Label')
        ax[i,j].plot(list(range(prod_period)),plot_label[:prod_period],'r-o',alpha=0.3,label= 'Production Data')
        ax[i,j].plot(list(range(prod_period,t_steps)),plotY_pred,'-',alpha=0.8, label='Prediction')
        ax[i,j].set_title('RMSE(ave): %.6f\n' % (testScore)+ ' Producing Time(month): %d' %(prod_period))
        ax[i,j].legend(loc='upper right',fontsize='small')
        ymax = np.max(plot_label)*1.6
        ax[i,j].set_ylim([0, ymax])
        n+=1
        
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.95, wspace=0.28, hspace=0.5)
