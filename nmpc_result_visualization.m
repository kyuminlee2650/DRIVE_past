clear all; close all;

% 변수 직접 선언 및 초기화
lw_ref = 2.0;
lw_1 = 1.5;
lw_0 = 1.5;
c_mode0 = [0.8 0.2 0.2]; 
title_fs = 14;
label_fs = 12;
tick_fs = 11;
leg_fs = 11;

load('nmpc_result.mat')

% ========================================================================
% FIGURE 1: 기본 궤적 플롯
% ========================================================================
figure('Name', '주행 궤적 비교 (Trajectory)', 'NumberTitle', 'off', 'Position', [100, 100, 750, 650]);
hold on; grid on; box on;

plot(res.isDRIVE_1.desired_X, res.isDRIVE_1.desired_Y, 'k-',  'LineWidth', lw_ref, 'DisplayName', 'Desired Path');
plot(res.isDRIVE_1.ego_X, res.isDRIVE_1.ego_Y, 'y-.',  'LineWidth', lw_1, 'DisplayName', 'DRIVE(Ours)');
plot(res.isDRIVE_0.ego_X, res.isDRIVE_0.ego_Y, 'r--', 'Color', c_mode0, 'LineWidth', lw_0, 'DisplayName', 'DDM(baseline)');

title('Vehicle Trajectories', 'FontSize', title_fs, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
xlabel('Global X [m]', 'FontSize', label_fs, 'FontName', 'Times New Roman');
ylabel('Global Y [m]', 'FontSize', label_fs, 'FontName', 'Times New Roman');

set(gca, 'FontSize', tick_fs, 'FontName', 'Times New Roman');

axis equal;
x_limits = xlim; y_limits = ylim;
x_span = x_limits(2) - x_limits(1); y_span = y_limits(2) - y_limits(1);
margin_ratio = 0.10; 
xlim([x_limits(1) - x_span * margin_ratio, x_limits(2) + x_span * margin_ratio]);
ylim([y_limits(1) - y_span * margin_ratio, y_limits(2) + y_span * margin_ratio]);

legend('Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', leg_fs, 'FontName', 'Times New Roman');

% ========================================================================
% FIGURE 2: 위험 구간 마킹 궤적
% ========================================================================
figure('Name', '주행 궤적 비교 (Trajectory)', 'NumberTitle', 'off', 'Position', [100, 100, 750, 700]);
hold on; grid on; box on;

plot(res.isDRIVE_1.desired_X, res.isDRIVE_1.desired_Y, 'k-',  'LineWidth', lw_ref, 'HandleVisibility', 'off');
plot(res.isDRIVE_1.ego_X, res.isDRIVE_1.ego_Y, 'y-.',  'LineWidth', lw_1, 'HandleVisibility', 'off');
plot(res.isDRIVE_0.ego_X, res.isDRIVE_0.ego_Y, 'r--',  'LineWidth', lw_0, 'HandleVisibility', 'off');

marker_spacing = 5; 

% DRIVE(Ours) 모드 마킹
idx_6m_1 = abs(res.isDRIVE_1.lateral_err) > 6.0;
if any(idx_6m_1)
    err_time_1 = res.isDRIVE_1.time_25Hz(idx_6m_1);
    high_err_X1 = interp1(res.isDRIVE_1.time_1000Hz, res.isDRIVE_1.ego_X, err_time_1, 'linear', 'extrap');
    high_err_Y1 = interp1(res.isDRIVE_1.time_1000Hz, res.isDRIVE_1.ego_Y, err_time_1, 'linear', 'extrap');
    plot(high_err_X1(1:marker_spacing:end), high_err_Y1(1:marker_spacing:end), ...
         'go', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'DRIVE Error > 6m');
end

% DDM(baseline) 모드 마킹
idx_6m_0 = abs(res.isDRIVE_0.lateral_err) > 6.0;
if any(idx_6m_0)
    err_time_0 = res.isDRIVE_0.time_25Hz(idx_6m_0);
    high_err_X0 = interp1(res.isDRIVE_0.time_1000Hz, res.isDRIVE_0.ego_X, err_time_0, 'linear', 'extrap');
    high_err_Y0 = interp1(res.isDRIVE_0.time_1000Hz, res.isDRIVE_0.ego_Y, err_time_0, 'linear', 'extrap');
    plot(high_err_X0(1:marker_spacing:end), high_err_Y0(1:marker_spacing:end), ...
         'mo', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'DDM Error > 6m');
end

title('Vehicle Trajectories', 'FontSize', title_fs, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
xlabel('Global X [m]', 'FontSize', label_fs, 'FontName', 'Times New Roman');
ylabel('Global Y [m]', 'FontSize', label_fs, 'FontName', 'Times New Roman');

set(gca, 'FontSize', tick_fs, 'FontName', 'Times New Roman');
axis equal;

x_limits = xlim; y_limits = ylim;
x_span = x_limits(2) - x_limits(1); y_span = y_limits(2) - y_limits(1);
xlim([x_limits(1) - x_span * 0.1, x_limits(2) + x_span * 0.1]);
ylim([y_limits(1) - y_span * 0.1, y_limits(2) + y_span * 0.1]);

legend('Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', leg_fs, 'FontName', 'Times New Roman');