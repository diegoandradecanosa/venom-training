import transformers
import torch


import venom_sparse_operations as venom_sp_ops









#def main(num_repeats, number):
def main():
    
    v = 64
    n = 2
    m = 8


    
    #torch.set_grad_enabled(False)# Cannot disable if we want to test backwards propagation.

    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-uncased')
    model = model.to(device='cuda:0').half()
    
    #Replace linear operations with SPMM kernel
    venom_sp_ops.linear_to_venom_spmm(model, v, n, m)
    
    
    # Prepare model input
    data = torch.randint(low=0, high=100, size=(32, 512))#, dtype=torch.half)
    data = data.to(device='cuda:0')
    labels = torch.randint(low=0, high=100, size=(32, 1024)).to(device='cuda:0')
    
    # Run model in forward direction
    
    prediction = model(data)
    
    #prediction.backwards()
    
    #print(prediction)
    logits = prediction[0] 
    #print("logits: " + str(logits))

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)

    
    # Run model in backwards direction
    #loss = (prediction - labels).sum()
    #loss.backward()
    loss.backward()
    
    print("")
    
    
if __name__ == "__main__":
    torch.set_grad_enabled(True)
    #main(num_repeats=100, number=1)
    main()