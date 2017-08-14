function plot_eigen(data)


num_tmp = 162;
data_l = zeros(num_tmp,2);  % cell(2,1);
data_r = zeros(num_tmp,2); %cell(2,1);

vec_l = zeros(num_tmp,2);
vec_r = zeros(num_tmp,2);

vec_total = cell(1,num_tmp*2);

for i = 1:num_tmp
    if(size(data{1}{i},1)>size(data{1}{i},2) )
        [tmp_vec_l,tmp_l] = eig(cov(data{1}{i})/trace(cov(data{1}{i})));
        [tmp_vec_r,tmp_r] = eig(cov(data{2}{i})/trace(cov(data{2}{i})));
    else
        [tmp_vec_l,tmp_l] = eig(cov(data{1}{i}')/trace(cov(data{1}{i}')));
        [tmp_vec_r,tmp_r] = eig(cov(data{2}{i}')/trace(cov(data{2}{i}')));
    end
    
    tmp_l_d = diag(tmp_l);
    tmp_r_d = diag(tmp_r);
    
    vec_total{i} = tmp_vec_l;
    vec_total{i+num_tmp}=tmp_vec_r;
    
    data_l(i,1)=tmp_l_d(end);
    data_l(i,2)=tmp_l_d(end-1);
    
    vec_l(i,1) = tmp_vec_l(1,end);
    vec_l(i,2)=tmp_vec_l(2,end);
   
     data_r(i,1) = tmp_r_d(end);
     data_r(i,2) = tmp_r_d(end-1);
     vec_r(i,1) = tmp_vec_r(1,end);
     vec_r(i,2) = tmp_vec_r(2,end);
end

figure
plot(data_l(:,1),data_l(:,2),'r*');

hold on
 plot(data_r(:,1),data_r(:,2),'ko');
 %legend('left','right');

 
 %data_all = [data_l(:,1);data_r(:,1)];
 %var(data_all)
 vec_all=[vec_l(:,1);vec_r(:,1)];
 var(vec_all)
 
 figure
 plot(vec_l(:,1),vec_l(:,2),'r*');
 hold on
 plot(vec_r(:,1),vec_r(:,2),'ko');
 save tmp_vector vec_total

