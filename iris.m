%==========================================================
% Load and prepare the Iris dataset for clustering/classification tasks.
% The script extracts selected features and encodes the class labels numerically.
%==========================================================

% Load the built-in Fisher Iris dataset
% Variables:
%   meas    - Numeric matrix (150x4) containing four features per sample:
%              [sepal length, sepal width, petal length, petal width]
%   species - Cell array (150x1) with species names: 'setosa', 'versicolor', 'virginica'
load fisheriris           

% Extract features matrix
X = meas;                  % Assign all features to X

% Select specific features: sepal length (1), petal length (3), and petal width (4)
X = [X(:,1) X(:,3) X(:,4)];
X = X ./ vecnorm(X, 2, 2);     % Renormalize to unit sphere

% Define the number of distinct classes (species)
q = 3;

% Convert species names to numeric labels (1=setosa, 2=versicolor, 3=virginica)
true_labels = grp2idx(species);
