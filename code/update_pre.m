function result = update_pre(gm,data_source,data_target)

result=gm;
n = 162;

%% initial 
for i = 1:162*2
    C_new = cov(data_target{floor((i-1)/n)+1}{mod((i-1),n)+1}')/trace(cov(data_target{floor((i-1)/n)+1}{mod((i-1),n)+1}'));
    
    dist_1 = sqrt(sum(sum((C_new-result{1}).^2)));
    dist_2 = sqrt(sum(sum((C_new-result{2}).^2)));
    theta_1 = 1-(dist_1)/(dist_1+dist_2);
    theta_2 = 1-(dist_2)/(dist_1+dist_2);
    
    result{1} = result{1}*n/(n+1) + C_new*theta_1/(1+n);
    result{2} = result{2}*n/(n+1) + C_new*theta_2/(1+n);
end
