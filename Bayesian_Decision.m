load Data_test(1).mat;
load Data_Train(1).mat;
load Label_Train(1).mat;

data_class = [1, 2, 3];
data_size = size(Data_Train);

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

data = dataTrain;
label = labelTrain;
num_class = length(data_class);

% train_class_idx identifies the data indices belonging to each class
% train_index shows the total number of each class in the data
[train_class_idx, train_index] = class_loc_idx(label, data_class);
[val_class_idx, val_index] = class_loc_idx(labelVal, data_class);

prior_prob = prior_probability(train_index);
[class_mean, class_covariance] = find_class_mean(data, train_class_idx, train_index);

val = predict(dataVal, num_class, class_mean, class_covariance, prior_prob);

validation_accuracy = score(val, labelVal)

test = predict(Data_test, num_class, class_mean, class_covariance, prior_prob);

function accuracy = score(predict, target)
    accuracy = 0;
    for i = 1:length(predict)
        if predict(i) == target(i)
            accuracy = accuracy + 1;
        end
    end
    accuracy = accuracy/length(predict);
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

function output = predict(data, num_class, class_mean, class_covariance, prior_prob)
    discriminant_fn = find_df(data, num_class, class_mean, class_covariance, prior_prob);
    output = zeros([length(discriminant_fn), 1]);
    for i = 1:length(output)
        largest_discriminant_fn = max(discriminant_fn(i, :));
        for j = 1:length(discriminant_fn(i, :))
            if discriminant_fn(i, j) == largest_discriminant_fn
                output(i) = j;
            end
        end
    end
end

function [class_mean, class_covariance] = find_class_mean(data, class_idx, index)
    class_mean = zeros([size(data, 2),length(index)]);
    class_covariance = zeros([size(data, 2),(size(data, 2)*length(index))]);
    for i = 1:length(index)
        class = class_idx(i, 1:index(i));
        data_class = data(class, :);
        mean = find_mean(data_class);
        covariance = find_covariance(data_class, mean);
        class_mean(:, i) = mean;
        class_covariance(:, (i - 1)*size(data, 2)+1:(i - 1)*size(data, 2)+4) = covariance;
    end
end

function mean_v = find_mean(data)
    mean_v = zeros([size(data, 2) 1]);
    for i = 1:length(mean_v)
        mean_v(i) = mean(data(:, i));
    end
end

function covariance_m = find_covariance(data, mean_v)
    covariance_m = zeros([size(data, 2) size(data, 2)]);
    for j = 1:length(data)
        covariance_m = covariance_m + (data(j, :)' - mean_v) * (data(j, :)' - mean_v)';
    end
    covariance_m = covariance_m/length(data);
end

function discriminant_fn = find_df(data, num_class, class_mean, class_covariance, prior_prob)
    discriminant_fn = zeros([size(data, 1) num_class]);
    for i = 1:length(discriminant_fn)
        for j = 1:num_class
            mean = class_mean(:, j);
            covariance = class_covariance(:, (j - 1)*size(data, 2)+1:(j - 1)*size(data, 2)+4);
            discriminant_fn(i, j) = class_con_prob(data(i, :), mean, covariance) ...
                * prior_prob(j);
        end
    end
end 

function prior_prob = prior_probability(data_index)
    prior_prob = zeros([length(data_index), 1]);
    sum = 0;
    for i = 1:length(data_index)
        sum = sum + data_index(i);
    end 
    for i = 1:length(data_index)
        prior_prob(i) = data_index(i)/sum;
    end 

end

function class_con_prob = class_con_prob(x, mean_v, covariance_m)
    class_con_prob = (1/((2*pi)^(length(mean_v))*det(covariance_m)^(1/2))) ...
        * exp(-0.5 * ((x' - mean_v)' * inv(covariance_m) * (x' - mean_v)));
end

function mean_v = find_mean_gmm(gamma, data)
    mean_v = zeros([size(data, 2) 1]);
    for i = 1:length(mean_v)
        mean_v(i) = mean(data(:, i));
    end
end