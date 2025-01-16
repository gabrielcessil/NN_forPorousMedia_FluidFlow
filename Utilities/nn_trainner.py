import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#### NEURAL NETWORK UTILITIES 

def train_one_epoch(model, train_loader, loss_function, optimizer):
    model.train()

    running_loss = 0.0    
    # Para cada batch
    for batch_inputs, batch_labels in train_loader:
        
        # REALIZA BATCH:
        optimizer.zero_grad() # Reinicia os gradientes para calculo do passo dos pesos
        batch_outputs = model(batch_inputs) # Calcula Saídas
        loss = loss_function(batch_outputs, batch_labels) # Calcula Custo
        loss.backward() # Calcula gradiente em relacao do Custo
        optimizer.step() # Realiza um passo dos pesos
        
        running_loss += loss.item() # Atualizar a perda acumulada da epoca

    # Perda média para a última época
    avg_loss = running_loss / len(train_loader)

    return avg_loss, model

def validate_one_epoch(model, valid_loader, loss_function):
    
    with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
    
      model.eval() # Entre em modo avaliacao, desabilitando funcoes exclusivas de treinamento (ex:Dropout)
      valid_summed_loss = 0.0

      # Para cada sample de validacao
      for valid_input, valid_label in valid_loader:
          # Gera a saida do modelo atual
          valid_output = model(valid_input)
          # Calcula o loss entre output gerado e label
          valid_loss = loss_function(valid_output, valid_label)
          valid_summed_loss += valid_loss.item()
         
      valid_avg_loss = valid_summed_loss / len(valid_loader)
      
      return valid_avg_loss
    

def full_train(model, train_loader, valid_loader, loss_function, optimizer,N_epochs=1000, file_name = "model_weights", include_time=False):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_cost_history = []
    val_cost_history = []
    best_valid_loss = np.inf    
    saved_model = None
    
    for epoch_index in range(N_epochs):
          
      print("\nExecutando epoca ", epoch_index, " /", N_epochs)
      train_avg_loss, model = train_one_epoch( 
          model=model,
          train_loader=train_loader,
          loss_function=loss_function, 
          optimizer=optimizer)
     
      valid_avg_loss = validate_one_epoch(
          model=model, 
          valid_loader=valid_loader,
          loss_function=loss_function)      
      
      print(' - Average Loss in train {}\n - Average Loss in validation {}'.format(train_avg_loss, valid_avg_loss))
      
      train_cost_history.append(train_avg_loss)
      val_cost_history.append(valid_avg_loss)
      
      # Track best performance, and save the model's state
      if valid_avg_loss < best_valid_loss:
          print("Melhor custo encontrado: ", valid_avg_loss, "\n")
          best_valid_loss = valid_avg_loss
          if include_time: model_path = file_name+"_{}_{}.pth".format(timestamp, epoch_index)
          else: model_path = file_name
          torch.save(model.state_dict(), model_path)
      
    return model_path, train_cost_history, val_cost_history


def Plot_Validation(train_cost_history, val_cost_history):
    
    # Trainning Visualization
    x = range(len(train_cost_history))
    plt.plot(x, train_cost_history, label='train', color='blue', marker='none')  # Line 1
    plt.plot(x, val_cost_history, label='validation', color='red', marker='none')     # Line 2
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cost History')
    plt.legend()
    plt.show()
    
    train_cost_rate_history = [train_cost_history[i+1]-train_cost_history[i] for i in range(len(train_cost_history)-1)]
    val_cost_rate_history = [val_cost_history[i+1]-val_cost_history[i] for i in range(len(val_cost_history)-1)]
    x = range(len(val_cost_rate_history))
    plt.plot(x, train_cost_rate_history, label='train', color='blue', marker='none')  # Line 1
    plt.plot(x, val_cost_rate_history, label='validation', color='red', marker='none')     # Line 2
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cost Rate History')
    plt.legend()
    plt.show()
