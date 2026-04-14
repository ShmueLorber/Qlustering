function [phi, tags] = Overlap3DWaveFunctionGenerator(q,num_states,width, centers)
% Generate quantum state vectors with width interpolation
% Inputs:
%   num_states - total number of states
%   q - number of groups
%   centers - q x 3 matrix of group centers
%   width - interpolation parameter (0=centered, 1=uniform)
% Outputs:
%   phi - state vectors (normalized)
%   tags - group labels

counts = round(num_states / q) .* ones(1, q);
phi = [];
tags = [];

for i = 1:q
    % Generate uniform random points on unit sphere
    uniform_points = randn(counts(i), 3);
    uniform_points = uniform_points ./ vecnorm(uniform_points, 2, 2);
    
    % Interpolate between center and uniform distribution
    center_normalized = centers(i,:) ./ norm(centers(i,:));
    interpolated_points = (1-width) * center_normalized + width * uniform_points;
    
    % Renormalize to unit sphere
    interpolated_points = interpolated_points ./ vecnorm(interpolated_points, 2, 2);
    
    phi = [phi; interpolated_points];
    tags = [tags; i.*ones(counts(i),1)];
end
     % Define custom colors for each group
    customColors = [
        0.9, 0.1, 0.1;  % Red
        0.1, 0.7, 0.2;  % Green
        0.1, 0.3, 0.9;  % Blue
        0.8, 0.6, 0.1;  % Yellow
        0.6, 0.1, 0.7;%;  % Purple
        0.2, 0.8, 0.8;  % Cyan
        0.9, 0.5, 0.1;  % Orange
        0.5, 0.5, 0.5   % Gray
    ];
    
    % Plot each group with the defined colors
    figure;
    scatter3(phi(:,1), phi(:,2), phi(:,3), 36, tags, 'filled');
    colormap(customColors);
    title('Normalized Quantum Data Using randcorr');
    xlabel('X'); ylabel('Y'); zlabel('Z');

end
