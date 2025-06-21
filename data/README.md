Put all csv files in here.

test_merged_final.csv:
id,date,store_nbr,family,onpromotion,dcoilwtico,city,state,cluster,is_working_day,payday,quake_severe,quake_moderate

train_merged_final.csv:
id,date,store_nbr,family,sales,onpromotion,dcoilwtico,city,state,cluster,is_working_day,payday,quake_severe,quake_moderate

test_merged_final_with_features.csv: 
id,date,store_nbr,family,onpromotion,dcoilwtico,city,state,cluster,is_working_day,payday,quake_severe,quake_moderate,year,month,day,dayofweek,quarter,month_sin,month_cos,dayofweek_sin,dayofweek_cos,store_nbr_encoded,family_encoded,onpromotion_log,onpromotion_scaled,dcoilwtico_scaled,city_encoded,state_encoded,cluster_encoded

train_merged_final_with_features.csv: 
id,date,store_nbr,family,sales,onpromotion,dcoilwtico,city,state,cluster,is_working_day,payday,quake_severe,quake_moderate,year,month,day,dayofweek,quarter,month_sin,month_cos,dayofweek_sin,dayofweek_cos,store_nbr_encoded,family_encoded,onpromotion_log,onpromotion_scaled,dcoilwtico_scaled,city_encoded,state_encoded,cluster_encoded
