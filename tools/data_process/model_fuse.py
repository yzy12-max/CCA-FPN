import collections


models = ''
new_model = ''
model_dict = [cur_model.state_dict() for cur_model in models]
paramter_name_list = list(model_dict[0].keys())
new_model_parameter = collections.OrderedDict()
for paramter_name in paramter_name_list:
    paramter_val = 0
    for i in range(len(model_dict)):
        paramter_val= paramter_val + model_dict[i][paramter_name]
    new_model_parameter[paramter_name]=paramter_val/len(model_dict)
new_model.load_state_dict(new_model_parameter)
