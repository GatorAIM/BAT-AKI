import torch
import torch.nn.functional as F

def compute_mlm_loss_without_weight(logits, labels, loss_fn, vocab_size):
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    return loss_fn(logits_flat, labels_flat)

def compute_mlm_loss_with_weight(logits, labels, loss_fn, vocab_size, token2weight):
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    return loss_fn(logits_flat, labels_flat, token2weight)
    
def compute_flag2_loss(hidden, attention_mask, flag2, classifier):
    mask = attention_mask.unsqueeze(-1).float()
    mean_hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)
    pred = classifier(mean_hidden).squeeze(-1)
    return F.binary_cross_entropy_with_logits(pred, flag2.float())


def forward_nomodule_noonto(model, batch, device):
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    delta_t = batch['delta_t'].to(device)
    segment_ids = batch['segment_ids'].to(device)
    logits, _, hidden = core_model(input_ids, attention_mask, delta_t, segment_ids)
    return logits, hidden, attention_mask


def forward_nomodule_onto(model, batch, device):
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    delta_t = batch['delta_t'].to(device)
    segment_ids = batch['segment_ids'].to(device)
    ontology_ids = batch['ontology_ids'].to(device)
    mrontology_ids = batch.get('mrontology_ids', None)                 
    if mrontology_ids is not None:
        mrontology_ids = mrontology_ids.to(device)                     
    logits, _, hidden = core_model(
        input_ids, attention_mask, delta_t, segment_ids,
        ontology_ids=ontology_ids,
        mrontology_ids=mrontology_ids                                   
    )
    return logits, hidden, attention_mask

def forward_nomodule_noonto_yesmrontology(model, batch, device):
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    delta_t = batch['delta_t'].to(device)
    segment_ids = batch['segment_ids'].to(device)
    mrontology_ids = batch['mrontology_ids'].to(device)  
    logits, _, hidden = core_model(
        input_ids, attention_mask, delta_t, segment_ids,
        ontology_ids=None,                           
        mrontology_ids=mrontology_ids                
    )
    return logits, hidden, attention_mask

def forward_nomodule_onto_nomrontology(model, batch, device):
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    delta_t = batch['delta_t'].to(device)
    segment_ids = batch['segment_ids'].to(device)
    ontology_ids = batch['ontology_ids'].to(device)      
    logits, _, hidden = core_model(
        input_ids, attention_mask, delta_t, segment_ids,
        ontology_ids=ontology_ids,                       
        mrontology_ids=None                              
    )
    return logits, hidden, attention_mask


def forward_module_noonto(model, batch, device):
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    delta_t = batch['delta_t'].to(device)
    segment_ids = batch['segment_ids'].to(device)
    module_ids = batch['module_ids'].to(device)
    logits, _, hidden = core_model(input_ids, attention_mask, delta_t, segment_ids, module_ids=module_ids)
    return logits, hidden, attention_mask


def forward_module_onto(model, batch, device):
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    delta_t = batch['delta_t'].to(device)
    segment_ids = batch['segment_ids'].to(device)
    module_ids = batch['module_ids'].to(device)
    ontology_ids = batch['ontology_ids'].to(device)
    
    mrontology_ids = batch.get('mrontology_ids', None)          
    if mrontology_ids is not None:                                   
        mrontology_ids = mrontology_ids.to(device)              

    logits, _, hidden = core_model(
        input_ids, attention_mask, delta_t, segment_ids,
        module_ids=module_ids,
        ontology_ids=ontology_ids,
        mrontology_ids=mrontology_ids                           
    )
    return logits, hidden, attention_mask


def evaluate_loss(model, val_loader, loss_fn, device, vocab_size=None, use_module_emb=False, use_ontology=False, use_mrontology=False, do_flag2=False, use_token2weight=False, token2weight=None):
    model.eval()
    model.eval()
    total_loss = 0
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    with torch.no_grad():
        for batch in val_loader:
            
            if use_module_emb and use_ontology:
                logits, hidden, attention_mask = forward_module_onto(model, batch, device)
            elif use_module_emb and not use_ontology:
                logits, hidden, attention_mask = forward_module_noonto(model, batch, device)
            elif (not use_module_emb) and use_ontology and (not use_mrontology):
                logits, hidden, attention_mask = forward_nomodule_onto_nomrontology(model, batch, device)
            elif (not use_module_emb) and (not use_ontology) and use_mrontology:
                logits, hidden, attention_mask = forward_nomodule_noonto_yesmrontology(model, batch, device)  
            elif not use_module_emb and use_ontology and use_mrontology:
                logits, hidden, attention_mask = forward_nomodule_onto(model, batch, device)
            else:
                logits, hidden, attention_mask = forward_nomodule_noonto(model, batch, device)

            labels = batch['labels'].to(device)
            if use_token2weight:
                mlm_loss = compute_mlm_loss_with_weight(logits, labels, loss_fn, vocab_size, token2weight)
            else:
                mlm_loss = compute_mlm_loss_without_weight(logits, labels, loss_fn, vocab_size)
           
            if do_flag2:
                flag2 = batch['flag2'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                flag_loss = compute_flag2_loss(hidden, attention_mask, flag2, core_model.flag_classifier)
                loss = mlm_loss + flag_loss
            else:
                loss = mlm_loss

            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            positions = torch.arange(0, input_ids.size(1), device=device).unsqueeze(0)
            x = model.embedding(input_ids) + model.pos_embedding(positions)
            x = model.dropout(x)
            
            for layer in model.encoder_layers:
                x, _ = layer(x, src_key_padding_mask=~attention_mask)

            cls_index = attention_mask.int().argmax(dim=1)  
            cls_vectors = x[torch.arange(x.size(0)), cls_index]  

            features.append(cls_vectors.cpu())

    return torch.cat(features, dim=0)  