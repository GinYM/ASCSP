function plot_power(data)

num_tmp = length(data{1});
data_l = zeros(num_tmp,1);
data_r = zeros(num_tmp,1);
for i=1:num_tmp
    data_l(i) = data{1}{i}(1);
    data_r(i) = data{2}{i}(2);
end

plot(1:num_tmp,data_l,'*');
hold on
plot(1:num_tmp,data_r,'o');
