load Data_test(1).mat;
load Data_Train(1).mat;
load Label_Train(1).mat;

data_class = [1, 2, 3];
data_size = size(Data_Train);
plot_marker = {'-ro', '-g*', '-bx'};
% Test-train split
rng('default');
% Cross validation (train: 70%, test: 30%)
cv = cvpartition(size(Data_Train,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
dataTrain = Data_Train(~idx,:);
labelTrain = Label_Train(~idx, :);
dataVal  = Data_Train(idx,:);
labelVal = Label_Train(idx, :);
dataTest = Data_test;

dataTrain = Data_Train;
labelTrain = Label_Train;
dataVal  = Data_Train;
labelVal = Label_Train;
dataTest = Data_test;

data = dataTrain;
label = labelTrain;
num_class = length(data_class);

% train_class_idx identifies the data indices belonging to each class
% train_index shows the total number of each class in the data
[train_class_idx, train_index] = class_loc_idx(label, data_class);
[val_class_idx, val_index] = class_loc_idx(labelVal, data_class);
class_mean = find_class_mean(data, train_class_idx, train_index);
total_mean = find_mean(data);
s_w = scatter_within(class_mean, data, train_class_idx, train_index);
s_b = scatter_between(class_mean, total_mean, train_index);
w = weights(s_w, s_b, num_class);

projected_mean_1 = projection(w, 1, data, train_class_idx, train_index, plot_marker);
w0_1 = -0.5 * (projected_mean_1(1) + projected_mean_1(3));
w_1 = w(:, 1);
projected_mean_2 = projection(w, 2, data, train_class_idx, train_index, plot_marker);
w0_2 = -0.5 * (projected_mean_2(1) + projected_mean_2(3));
w_2 = w(:, 2);

val = find_df(dataVal, w_1, w0_1, w_2, w0_2);
validation_accuracy = score(val, labelVal)

test = find_df(dataTest, w_1, w0_1, w_2, w0_2);
% figure
% g1 = zeros([length(data), 1]);
% idx = zeros([length(data), 1]);
% for i = 1:length(data)
%     idx(i) = i;
%     g1(i) = w(:, 1)'*data(i, :)';
% end
% 
% class1 = train_class_idx(1, 1:train_index(1));
% class2 = train_class_idx(2, 1:train_index(2));
% class3 = train_class_idx(3, 1:train_index(3));
% 
% plot(idx(1:length(class1)), g1(class1),'-o');
% hold on;
% plot(idx(length(class1) + 1:length(class1) + length(class2)), g1(class2),'-*');
% plot(idx(length(class1) + length(class2) + 1:length(class1) + length(class2) + length(class3)), g1(class3),'-+');
% ------------------------------------------
projected_x = zeros([length(dataVal), 2]);
for i = 1:length(projected_x)
    projected_x(i, 1) = w(:, 1)'*dataVal(i, :)'+ w0_1;
    projected_x(i, 2) = w(:, 2)'*dataVal(i, :)'+ w0_2;
end
linspace_x = linspace(-3, 3, 100)';
linspace_y = linspace(-1, 1, 100)';
hold off;
figure;
hold on;
point_width = 5;
for i = 1:length(linspace_x)
    for j = 1:length(linspace_y)
        g1 = linspace_x(i);
        g2 = linspace_y(j);
        if g1 < 0 && g2 < 0
            plot(g1, g2, 's', 'MarkerSize', point_width, 'MarkerEdgeColor','yellow', 'MarkerFaceColor', 'yellow');
        elseif g1 > 0 && g2 > 0
            plot(g1, g2, 's', 'MarkerSize', point_width, 'MarkerEdgeColor','cyan', 'MarkerFaceColor', 'cyan');
        elseif g1 > 0 && g2 < 0
            plot(g1, g2, 's', 'MarkerSize', point_width, 'MarkerEdgeColor','magenta', 'MarkerFaceColor', 'magenta');
        else
            if g1 < g2
                plot(g1, g2, 's', 'MarkerSize', point_width, 'MarkerEdgeColor','yellow', 'MarkerFaceColor', 'yellow');
            elseif g1 > g2
                plot(g1, g2, 's', 'MarkerSize', point_width, 'MarkerEdgeColor','cyan', 'MarkerFaceColor', 'cyan');
            end
        end
    end
end
for i = 1:length(val_index)
    class = val_class_idx(i, 1:val_index(i));
    projected_x_class = projected_x(class, :);
    plot(projected_x_class(:, 1), projected_x_class(:, 2), plot_marker{i}(2:3));
end
xlabel('g1(x)');
ylabel('g2(x)');
    
xlim([-3 2]) ;
ylim([-0.6 1]);

function accuracy = score(predict, target)
    accuracy = 0;
    for i = 1:length(predict)
        if predict(i) == target(i)
            accuracy = accuracy + 1;
        end
    end
    accuracy = accuracy/length(predict);
end

function output = find_df(data, w_1, w0_1, w_2, w0_2)
    output = zeros([length(data), 1]);
    for i = 1:length(data)
        data_i = data(i, :)';
        g1 = w_1'*data_i + w0_1;
        g2 = w_2'*data_i + w0_2;

        if g1 > 0 && g2 > 0
            output(i) = 3;
        elseif g1 < 0 && g2 < 0
            output(i) = 1;
        elseif g1 > 0 && g2 < 0
            output(i) = 2;
        else
            if g1 > g2
                output(i) = 3;
            elseif g1 < g2
                output(i) = 1;
            end
        end
    end 
end

function projected_mean = projection(weights, weight_idx, data, class_idx, index, plot_marker)
    wTx = zeros([length(data), 1]);
    idx = zeros([length(data), 1]);
    projected_mean = zeros([length(index), 1]);
    for i = 1:length(data)
        idx(i) = i;
        wTx(i) = weights(:, weight_idx)'*data(i, :)';
    end
    figure;
    hold on;
    for i = 1:length(index)
        class = class_idx(i, 1:index(i));
        wTx_i = wTx(class);
        projected_mean(i) = mean(wTx_i);
        
        if i == 1
            idx_i = idx(1:index(i));
        else
            idx_cumulative = 0;
            for j = 1:(i - 1)
                idx_cumulative = idx_cumulative + index(j);
            end
            idx_i = idx(idx_cumulative + 1: idx_cumulative + index(i));
        end
        mean_y = ones([length(idx_i), 1])*projected_mean(i);

        plot(idx_i, wTx_i, plot_marker{i}, 'DisplayName',['class ', int2str(i)]);
        plot(idx_i, mean_y, plot_marker{i}(1:2), 'DisplayName',['class ', int2str(i), ' projected mean']);
    end
    legend;
    xlabel('index');
    ylabel(['Projection on w', int2str(weight_idx)]);
    hold off;
end

function w = weights(s_w, s_b, num_class)
    [v, d] = eig(inv(s_w)*s_b);
%     [v, d] = eig(s_b, s_w);
    [lambda, ind] = sort(diag(d), 'descend');
    v
    d
    w = v(:, [ind(1:num_class-1)])
end

function [class_idx, index] = class_loc_idx(labelData, dataClass)
    class_idx = zeros(length(dataClass), length(labelData));
    index = zeros([length(dataClass), 1]);
    for i = 1:length(labelData)
        for j = 1:length(dataClass)
            if labelData(i) == dataClass(j)
                index(j) = index(j) + 1;
                class_idx(j, index(j)) = i;
            end
        end
    end
end

function class_mean = find_class_mean(data, class_idx, index)
    class_mean = zeros([size(data, 2),length(index)]);
    for i = 1:length(index)
        class = class_idx(i, 1:index(i));
        data_class = data(class, :);
        mean = find_mean(data_class);
        class_mean(:, i) = mean;
    end
end

function mean_v = find_mean(data)
    mean_v = zeros([size(data, 2) 1]);
    for i = 1:length(mean_v)
        mean_v(i) = mean(data(:, i));
    end
end

function s_w = scatter_within(class_mean, data, class_idx, index)
    s_w = zeros([size(data, 2),size(data, 2)]);
    for i = 1:length(index)
        mean = class_mean(:, i);
        class = class_idx(i, 1:index(i));
        data_class = data(class, :);
        s_i = zeros([size(data_class, 2),size(data_class, 2)]);
        for j = 1:size(data_class, 1)
            x = data_class(j, :)'
            s_i = s_i + ((x - mean)*(x - mean)');
        end
        s_w = s_w + s_i;
    end
end

function s_b = scatter_between(class_mean, mean, index)
    s_b = zeros([length(mean), length(mean)]);
    for i = 1:length(index)
        n_i = index(i);
        class_mean_i = class_mean(:, i);
        s_b = s_b + (n_i * ((class_mean_i - mean)*(class_mean_i - mean)'));
    end
end