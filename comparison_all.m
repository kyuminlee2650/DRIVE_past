%% =========================================================================
% Deep Dynamics Models - 다중 결과 파일(.mat) 통합 비교 시각화 스크립트
% 설명: 각 모델별 평가 결과 구조체를 로드하여 2x3 Tiled Layout으로 통합 비교 분석
%% =========================================================================
clear all; clear vars; close all; clc;

%% 1. 비교 대상 파일 및 범례 설정
% 분석하고자 하는 파일의 전체 경로 또는 파일명을 입력하세요.
% file_list = { ...
%     'EvalResults_Baseline_DDM_carsim.mat', ...
%     'EvalResults_DRIVE_carsim.mat' ...
% };
% mode = 'Simulation';

file_list = { ...
    'EvalResults_Baseline_DDM_real.mat', ...
    'EvalResults_DRIVE_real.mat' ...
};

mode='Real';

% 그래프 범례(Legend)에 표시할 각 모델의 이름을 순서대로 입력하세요.
model_labels = { ...
    'DDM', ...
    'DRIVE' ...
};

num_models = length(file_list);
if num_models ~= length(model_labels)
    error('파일 개수와 모델 라벨의 개수가 일치하지 않습니다.');
end

%% 2. 글꼴 서체 및 크기 설정
fn        = 'Times New Roman'; 
fs_title  = 15;                
fs_label  = 13;                
fs_legend = 11;                
fs_tick   = 11;                

%% 3. 데이터 로드
model_data = cell(1, num_models);
for m_idx = 1:num_models
    current_file = file_list{m_idx};
    if exist(current_file, 'file')
        loaded_struct = load(current_file);
        if isfield(loaded_struct, 'eval_metrics')
            model_data{m_idx} = loaded_struct.eval_metrics;
        end
    end
end

line_styles = {'-', '--', ':', '-.'};
markers = {'o', 's', '^', 'd'};
lw = 2.0; ms = 6;   

%% =========================================================================
%% Figure 생성 및 2x3 Tiled Layout 설정
%% =========================================================================
% 한 화면에 6개의 그래프를 여유롭게 담기 위해 Figure 크기 확장
figure('Name', 'Comprehensive Model Comparison', 'Position', [100, 100, 1400, 600]); 
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

h_plots_global = []; % 통합 범례를 위한 핸들 배열
active_labels_global = {}; % 통합 범례용 라벨

%% [Tile 1] 좌측 상단: FDE 트렌드 및 ADE 요약 박스
nexttile(t); hold on;
ade_summary_text = {'\textbf{ADE}'}; 

for m_idx = 1:num_models
    if isempty(model_data{m_idx}), continue; end
    metrics = model_data{m_idx};
    
    if isfield(metrics, 'fde_step_trend')
        x_axis = if_exist_axis(metrics, 'x_p', 2:26);
        opt = [line_styles{mod(m_idx-1, 4)+1}, markers{mod(m_idx-1, 4)+1}];
        
        p = plot(x_axis, metrics.fde_step_trend, opt, 'LineWidth', lw, 'MarkerSize', ms);
        
        % 범례용 핸들 수집 (첫 번째 그래프에서 한 번만 수집)
        h_plots_global(end+1) = p;
        active_labels_global{end+1} = model_labels{m_idx};
    end
    
    % ADE 요약 텍스트 수집 
    m_name = model_labels{m_idx};
    ade_summary_text{end+1} = sprintf('%s: %.4f m', m_name, metrics.ade_step_trend(25));
end
grid on; box on; set(gca, 'FontName', fn, 'FontSize', fs_tick); 
title(sprintf('\\textbf{%s : Average FDE}', mode), 'FontName', fn, 'FontSize', fs_title, 'Interpreter', 'latex');
xlabel('Prediction Horizon Step', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 
ylabel('Error [m]', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 

% ADE 요약 텍스트 상자 배치
text(0.05, 0.82, ade_summary_text, 'Units', 'normalized', ...
    'FontName', fn, 'FontSize', fs_tick-1, 'Interpreter', 'latex', ...
    'BackgroundColor', [1 1 1 0.85], 'EdgeColor', '#D3D3D3', 'LineWidth', 1);

%% [Tile 2] 중앙 상단: ax RMSE
nexttile(t); hold on;
for m_idx = 1:num_models
    if isempty(model_data{m_idx}), continue; end
    metrics = model_data{m_idx};
    x_axis = if_exist_axis(metrics, 'x_a', 0:24);
    opt = [line_styles{mod(m_idx-1, 4)+1}, markers{mod(m_idx-1, 4)+1}];
    plot(x_axis, metrics.rmse_a_step(:, 1), opt, 'LineWidth', lw, 'MarkerSize', ms);
end
grid on; box on; set(gca, 'FontName', fn, 'FontSize', fs_tick); 
title(sprintf('\\textbf{%s} : $a_x$ \\textbf{RMSE}', mode), 'FontName', fn, 'FontSize', fs_title, 'Interpreter', 'latex');
xlabel('Prediction Step', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 
ylabel('RMSE [$\mathrm{m/s^2}$]', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 

%% [Tile 3] 우측 상단: ay RMSE
nexttile(t); hold on;
for m_idx = 1:num_models
    if isempty(model_data{m_idx}), continue; end
    metrics = model_data{m_idx};
    x_axis = if_exist_axis(metrics, 'x_a', 0:24);
    opt = [line_styles{mod(m_idx-1, 4)+1}, markers{mod(m_idx-1, 4)+1}];
    plot(x_axis, metrics.rmse_a_step(:, 2), opt, 'LineWidth', lw, 'MarkerSize', ms);
end
grid on; box on; set(gca, 'FontName', fn, 'FontSize', fs_tick); 
title(sprintf('\\textbf{%s} : $a_y$ \\textbf{RMSE}', mode), 'FontName', fn, 'FontSize', fs_title, 'Interpreter', 'latex');
xlabel('Prediction Step', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 
ylabel('RMSE [$\mathrm{m/s^2}$]', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 

%% [Tile 4] 좌측 하단: vx RMSE
nexttile(t); hold on;
for m_idx = 1:num_models
    if isempty(model_data{m_idx}), continue; end
    metrics = model_data{m_idx};
    x_axis = if_exist_axis(metrics, 'x_s', 1:25);
    opt = [line_styles{mod(m_idx-1, 4)+1}, markers{mod(m_idx-1, 4)+1}];
    plot(x_axis, metrics.rmse_s_step(:, 1), opt, 'LineWidth', lw, 'MarkerSize', ms);
end
grid on; box on; set(gca, 'FontName', fn, 'FontSize', fs_tick); 
title(sprintf('\\textbf{%s} : $v_x$ \\textbf{RMSE}', mode), 'FontName', fn, 'FontSize', fs_title, 'Interpreter', 'latex');
xlabel('Prediction Step', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 
ylabel('RMSE [m/s]', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 

%% [Tile 5] 중앙 하단: vy RMSE
nexttile(t); hold on;
for m_idx = 1:num_models
    if isempty(model_data{m_idx}), continue; end
    metrics = model_data{m_idx};
    x_axis = if_exist_axis(metrics, 'x_s', 1:25);
    opt = [line_styles{mod(m_idx-1, 4)+1}, markers{mod(m_idx-1, 4)+1}];
    plot(x_axis, metrics.rmse_s_step(:, 2), opt, 'LineWidth', lw, 'MarkerSize', ms);
end
grid on; box on; set(gca, 'FontName', fn, 'FontSize', fs_tick); 
title(sprintf('\\textbf{%s} : $v_y$ \\textbf{RMSE}', mode), 'FontName', fn, 'FontSize', fs_title, 'Interpreter', 'latex');
xlabel('Prediction Step', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 
ylabel('RMSE [m/s]', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 

%% [Tile 6] 우측 하단: dpsi (Yaw Rate) RMSE
nexttile(t); hold on;
for m_idx = 1:num_models
    if isempty(model_data{m_idx}), continue; end
    metrics = model_data{m_idx};
    x_axis = if_exist_axis(metrics, 'x_s', 1:25);
    opt = [line_styles{mod(m_idx-1, 4)+1}, markers{mod(m_idx-1, 4)+1}];
    plot(x_axis, metrics.rmse_s_step(:, 3), opt, 'LineWidth', lw, 'MarkerSize', ms);
end
grid on; box on; set(gca, 'FontName', fn, 'FontSize', fs_tick); 
title(sprintf('\\textbf{%s} : $\\dot{\\psi}$ \\textbf{RMSE}', mode), 'FontName', fn, 'FontSize', fs_title, 'Interpreter', 'latex');
xlabel('Prediction Step', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 
ylabel('RMSE [rad/s]', 'FontName', fn, 'FontSize', fs_label, 'Interpreter', 'latex'); 

%% [Global Legend] 통합 범례 하단 배치
if ~isempty(h_plots_global)
    lgd = legend(h_plots_global, active_labels_global, ...
        'Orientation', 'horizontal', ...
        'FontName', fn, ...
        'FontSize', fs_legend, ...
        'Interpreter', 'latex');
    lgd.Layout.Tile = 'south'; % 창 맨 밑 테두리에 레전드를 바짝 붙입니다.
end

%% =========================================================================
%% 보조 함수: 축 데이터 누락 대응용 내부 매핑 함수
%% =========================================================================
function x_out = if_exist_axis(metrics_struct, field_str, default_range)
    if isfield(metrics_struct, field_str)
        x_out = metrics_struct.(field_str);
    else
        x_out = default_range;
    end
end