%adaptive learning with covariance shift
function vis_sa()
clear
close all
load store_plot

ms = 6;
plot_size = 3;

figure

num = 162;
select = 80;
% X = [ mvnrnd([0.5 1.5], [0.025 0.03 ; 0.03 0.16], num) ; ...
%       mvnrnd([1 1], [0.09 -0.01 ; -0.01 0.08], num)   ];
G = [1*ones(select,1) ; 2*ones(select,1)];

X_source = [X(1:select,1:2);X(num+1:select+num,1:2)];

s1 = plot(X_source(1:select,1),X_source(1:select,2),'b+','MarkerSize',ms);
hold on
s2 = plot(X_source(select+1:select*2,1),X_source(select+1:select*2,2),'bo','MarkerSize',ms);
%gscatter(X(1:2*num,1), X(1:2*num,2), G)
%axis equal, hold on
hold on


for k=1:2
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X_source(idx,1:2) );
    X0 = bsxfun(@minus, X_source(idx,1:2), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = V*sqrt(D);               %# scale eigenvectors
    
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space

    %# plot cov and major/minor axes
    if k == 1
        plot(e(1,:), e(2,:),'b','LineWidth',plot_size);
    else
        plot(e(1,:), e(2,:),'b','LineWidth',plot_size);
    end
    hold on
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end

X_target = [X(2*num+1:2*num+select,1:2);X(3*num+1:3*num+select,1:2)];

hold on
t1 = plot(X_target(1:select,1),X_target(1:select,2),'r+','MarkerSize',ms);
hold on
t2 = plot(X_target(select+1:2*select,1),X_target(select+1:2*select,2),'ro','MarkerSize',ms);
%gscatter(X(1:2*num,1), X(1:2*num,2), G)
%axis equal, hold on
hold on

xlim([-5.5,-1])
ylim([-3.5,-0.5])
%axis equal


for k=1:2
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X_target(idx,1:2) );
    X0 = bsxfun(@minus, X_target(idx,1:2), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = V*sqrt(D);               %# scale eigenvectors
    size(VV)
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space

    %# plot cov and major/minor axes
    if k == 1
        plot(e(1,:), e(2,:),'r','LineWidth',plot_size);
    else
        plot(e(1,:), e(2,:),'r','LineWidth',plot_size);
    end
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end


set(gca,'YTick',[-3.5 : 1: 1.5]);
set(gca,'XTick',[-5.5 : 1: 1.5]);
legend([s1,s2,t1,t2],'Source1','Source2','Target1','Target2');
set(gca,'fontname','times')
set(gca,'FontSize',12)

figure

num = 162;
select = 80;
% X = [ mvnrnd([0.5 1.5], [0.025 0.03 ; 0.03 0.16], num) ; ...
%       mvnrnd([1 1], [0.09 -0.01 ; -0.01 0.08], num)   ];
G = [1*ones(select,1) ; 2*ones(select,1)];

X_source = [Xproj(1:select,1:2);Xproj(num+1:select+num,1:2)];

s1 = plot(X_source(1:select,1),X_source(1:select,2),'b+','MarkerSize',ms);
hold on
s2 = plot(X_source(select+1:select*2,1),X_source(select+1:select*2,2),'bo','MarkerSize',ms);
hold on


for k=1:2
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X_source(idx,1:2) );
    X0 = bsxfun(@minus, X_source(idx,1:2), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = V*sqrt(D);               %# scale eigenvectors
    
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space

    %# plot cov and major/minor axes
    if k == 1
        plot(e(1,:), e(2,:),'b','LineWidth',plot_size);
    else
        plot(e(1,:), e(2,:),'b','LineWidth',plot_size);
    end
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end

X_target = [Xproj(2*num+1:2*num+select,1:2);Xproj(3*num+1:3*num+select,1:2)];

hold on
t1 = plot(X_target(1:select,1),X_target(1:select,2),'r+','MarkerSize',ms);
hold on
t2 = plot(X_target(select+1:2*select,1),X_target(select+1:2*select,2),'ro','MarkerSize',ms);
%gscatter(X(1:2*num,1), X(1:2*num,2), G)
%axis equal, hold on
hold on


for k=1:2
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X_target(idx,1:2) );
    X0 = bsxfun(@minus, X_target(idx,1:2), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = V*sqrt(D);               %# scale eigenvectors
    size(VV)
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space

    %# plot cov and major/minor axes
    if k == 1
        plot(e(1,:), e(2,:),'r','LineWidth',plot_size);
    else
        plot(e(1,:), e(2,:),'r','LineWidth',plot_size);
    end
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end

xlim([-2,2])
ylim([-1.5,1.5])
%axis equal
set(gca,'YTick',[-1.5 : 1: 1.5]);
set(gca,'fontname','times')
legend([s1,s2,t1,t2],'Source1','Source2','Target1','Target2');

set(gca,'FontSize',12)







figure
load previous_X
num = 162;
select = 80;
% X = [ mvnrnd([0.5 1.5], [0.025 0.03 ; 0.03 0.16], num) ; ...
%       mvnrnd([1 1], [0.09 -0.01 ; -0.01 0.08], num)   ];
G = [1*ones(select,1) ; 2*ones(select,1)];

X_source = [X(1:select,1:2);X(num+1:select+num,1:2)];

s1 = plot(X_source(1:select,1),X_source(1:select,2),'b+','MarkerSize',ms);
hold on
s2 = plot(X_source(select+1:select*2,1),X_source(select+1:select*2,2),'bo','MarkerSize',ms);
%gscatter(X(1:2*num,1), X(1:2*num,2), G)
%axis equal, hold on
hold on


for k=1:2
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X_source(idx,1:2) );
    X0 = bsxfun(@minus, X_source(idx,1:2), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = V*sqrt(D);               %# scale eigenvectors
    
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space

    %# plot cov and major/minor axes
    if k == 1
        plot(e(1,:), e(2,:),'b','LineWidth',plot_size);
    else
        plot(e(1,:), e(2,:),'b','LineWidth',plot_size);
    end
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end

X_target = [X(2*num+1:2*num+select,1:2);X(3*num+1:3*num+select,1:2)];

hold on
t1 = plot(X_target(1:select,1),X_target(1:select,2),'r+','MarkerSize',ms);
hold on
t2 = plot(X_target(select+1:2*select,1),X_target(select+1:2*select,2),'ro','MarkerSize',ms);
%gscatter(X(1:2*num,1), X(1:2*num,2), G)
%axis equal, hold on
hold on

% xlim([-5.5,-1])
% ylim([-3.5,-0.5])
%axis equal


for k=1:2
    %# indices of points in this group
    idx = ( G == k );

    %# substract mean
    Mu = mean( X_target(idx,1:2) );
    X0 = bsxfun(@minus, X_target(idx,1:2), Mu);

    %# eigen decomposition [sorted by eigen values]
    [V D] = eig( X0'*X0 ./ (sum(idx)-1) );     %#' cov(X0)
    [D order] = sort(diag(D), 'descend');
    D = diag(D);
    V = V(:, order);

    t = linspace(0,2*pi,100);
    e = [cos(t) ; sin(t)];        %# unit circle
    VV = V*sqrt(D);               %# scale eigenvectors
    size(VV)
    e = bsxfun(@plus, VV*e, Mu'); %#' project circle back to orig space

    %# plot cov and major/minor axes
    if k == 1
        plot(e(1,:), e(2,:),'r','LineWidth',plot_size);
    else
        plot(e(1,:), e(2,:),'r','LineWidth',plot_size);
    end
    hold on
    %#quiver(Mu(1),Mu(2), VV(1,1),VV(2,1), 'Color','k')
    %#quiver(Mu(1),Mu(2), VV(1,2),VV(2,2), 'Color','k')
end

xlim([-6,-1])
ylim([-4,0])
set(gca,'YTick',-4 : 1: 0);
set(gca,'XTick',-6 : 1: -1);
legend([s1,s2,t1,t2],'Source1','Source2','Target1','Target2');
set(gca,'fontname','times')
set(gca,'FontSize',12)
%axis equal
