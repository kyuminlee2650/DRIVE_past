%% =========================================================================
% Deep Dynamics Model (9-Dim Input) - 다중 CSV 배치 평가 및 시각화 스크립트
% 수정 사항: 각 물리량별 1-Step 및 25-Step 최대 절대 오차(Max Error) 연산 추가
%% =========================================================================
clear vars;      
clear functions; % persistent 변수 초기화를 위해 필수 호출
clear all; close all;

%% 1. 모델 경로 및 테스트 데이터 폴더 설정
model_path = "C:\Users\mumu2\Downloads\DRIVE_carsim_trial18.mat";
csv_folder = "C:\workspace\carsim_processed_data_test_25hz";

fprintf('모델 로드 중: %s\n', model_path);
model_data = load(model_path);

%% 0. ⚙️ [하이퍼파라미터 튜닝 선언부] - 이곳에서 모델 구조를 제어합니다.
tuning.seq_len        = model_data.opt_seq_len;       % 시퀀스 길이
tuning.hidden_dim     = model_data.opt_hidden_dim;     % 은닉층(Hidden) 차원
tuning.num_gru_layers = model_data.opt_num_layers;     % GRU 레이어 개수
tuning.num_fc_layers  = model_data.opt_num_fc_layers;  % MLP 백본 내 반복 추가 횟수
tuning.act_type       = model_data.opt_act_func;       % 활성화 함수 문자열

csv_files = dir(fullfile(csv_folder, '*.csv'));
num_files = length(csv_files);

if num_files == 0
    err_msg = sprintf('지정된 폴더에 CSV 파일이 존재하지 않습니다: %s', csv_folder);
    disp(err_msg);
    return;
end
fprintf('총 %d개의 테스트 CSV 파일을 발견했습니다. 평가를 시작합니다...\n\n', num_files);

%% 2. 차량 파라미터 기본 설정
params.dt = 0.04; 
params.m = 1270;
params.lf = 1.015; 
params.lr = 1.895;
params.seq_len = tuning.seq_len; 

% 네트워크 출력 Boundary 파라미터
params.t_min = model_data.t_min;
params.t_max = model_data.t_max;
params.dr_min = model_data.dr_min;
params.dr_max = model_data.dr_max;
params.Iz_min = model_data.Iz_min;
params.Iz_max = model_data.Iz_max;

%% 3. 글로벌 누적 오차 및 최대 오차 변수 초기화
total_sq_err_1_s = zeros(1, 3); 
total_sq_err_1_p = zeros(1, 3); 
total_sq_err_1_a = zeros(1, 2);
total_1_pts = 0;

step_horizon = 25; 
% 스텝별 오차를 누적하기 위해 25 x Dim 크기 초기화
total_sq_err_s_step = zeros(step_horizon, 3); 
total_sq_err_p_step = zeros(step_horizon, 3); 
total_sq_err_a_step = zeros(step_horizon, 2);

% 스텝별 Displacement 오차 누적용 변수
total_disp_err_step = zeros(step_horizon, 1);
total_ade_err = 0;
total_fde_err = 0;
total_seq_count = 0; 

% --- 최대 절대 오차(Max Absolute Error) 추적 변수 초기화 ---
max_err_1_s = zeros(1, 3); % [vx, vy, r]
max_err_1_p = zeros(1, 3); % [X, Y, psi]
max_err_1_a = zeros(1, 2); % [ax, ay]

max_err_25_s = zeros(1, 3);
max_err_25_p = zeros(1, 3);
max_err_25_a = zeros(1, 2);
max_err_fde  = 0;          % 최종 변위 오차 최댓값

%% 4. 다중 파일 순회 및 시뮬레이션 수행
for f_idx = 1:num_files
    csv_path = fullfile(csv_folder, csv_files(f_idx).name);
    fprintf('[%d/%d] 파일 처리 중: %s ... ', f_idx, num_files, csv_files(f_idx).name);
    
    data = readmatrix(csv_path, 'NumHeaderLines', 1);
    num_samples = size(data, 1) - params.seq_len - 2; 
    
    if num_samples < 1
        fprintf('데이터가 너무 짧아 건너뜁니다.\n');
        continue;
    end
    
    % 데이터 추출
    time      = data(:, 1);             
    pos_gt    = data(:, 2:4);         
    states_gt = data(:, 5:7);      
    ctrls     = data(:, 8:10);         
    accels_gt = data(:, 11:12);    
    
    % 로컬 버퍼 초기화
    preds_pinn_s = zeros(num_samples, 3); 
    preds_pinn_p = zeros(num_samples, 3); 
    preds_pinn_a = zeros(num_samples, 2);
    
    gt_s_compare = zeros(num_samples, 3); 
    gt_p_compare = zeros(num_samples, 3); 
    gt_a_compare = zeros(num_samples, 2); 
    
    multi_count = 0; 
    multi_idx_list = [];
    multi_pinn = struct('s', {}, 'p', {}, 'a', {}); 
    
    % 1-Step 및 25-Step 평가
    for i = 1:num_samples
        idx = i + params.seq_len - 1; 
        
        curr_t = time(idx);
        curr_s = states_gt(idx, :); 
        curr_p = pos_gt(idx, :); 
        curr_c = ctrls(idx, :);
        next_c = ctrls(idx+1, :); 
        
        % --- 9차원 로컬 좌표계 시퀀스 추출 ---
        seq_pos = pos_gt(i:idx, :);
        seq_states = states_gt(i:idx, :);
        seq_ctrls = ctrls(i:idx, :);
        
        mid_idx = floor(params.seq_len / 2) + 1;
        ref_pos = seq_pos(mid_idx, :);
        
        loc_pos_seq = global_to_local_matlab(seq_pos, ref_pos);
        in_unnorm = [loc_pos_seq, seq_states, seq_ctrls]; 
        input_seq = (in_unnorm - model_data.in_mean) ./ (model_data.in_std + 1e-6);
        
        % --- 1. 1-Step 예측 ---
        [tf, tr, dr, Iz_pred, ~] = get_pinn_params_stateful(input_seq, model_data, params, tuning, []);
        
        % 물리 스텝 실행
        [p_s_t1, p_p_t1, p_a_t] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz_pred, params);
        [~, p_p_t2, ~] = physics_step(p_s_t1, p_p_t1, next_c, tf, tr, dr, Iz_pred, params);
        
        preds_pinn_s(i, :) = p_s_t1; 
        preds_pinn_p(i, :) = p_p_t2; 
        preds_pinn_a(i, :) = p_a_t;
        
        % GT 저장
        gt_s_compare(i, :) = states_gt(idx+1, :);
        gt_p_compare(i, :) = pos_gt(idx+2, :);
        gt_a_compare(i, :) = accels_gt(idx, :);
        
        % --- 1-Step 최대 절대 오차 실시간 갱신 ---
        max_err_1_s = max([max_err_1_s; abs(p_s_t1 - states_gt(idx+1, :))], [], 1);
        max_err_1_p = max([max_err_1_p; abs(p_p_t2 - pos_gt(idx+2, :))], [], 1);
        max_err_1_a = max([max_err_1_a; abs(p_a_t - accels_gt(idx, :))], [], 1);
        
        % --- 2. 25-Step Recursive 예측 ---
        if mod(i, step_horizon) == 0 && curr_t >= 1.0 && (idx + step_horizon + 1) <= size(data, 1)
            multi_count = multi_count + 1;
            multi_idx_list(multi_count) = idx;
            
            ps_tmp = curr_s; pp_tmp = curr_p; 
            
            p_m_s = zeros(step_horizon, 3); 
            p_m_p = zeros(step_horizon, 3); 
            p_m_a = zeros(step_horizon, 2);
            
            for h_step = 1:(step_horizon + 1)
                f_idx_h = idx + h_step - 1; 
                f_ctrl = ctrls(f_idx_h, :);
                
                [ps_tmp, pp_tmp, pa_tmp] = physics_step(ps_tmp, pp_tmp, f_ctrl, tf, tr, dr, Iz_pred, params);
                
                if h_step <= step_horizon
                    p_m_s(h_step, :) = ps_tmp;
                    p_m_a(h_step, :) = pa_tmp;
                end
                if h_step > 1
                    p_m_p(h_step-1, :) = pp_tmp; 
                end
            end
            
            multi_pinn(multi_count).s = p_m_s; 
            multi_pinn(multi_count).p = p_m_p; 
            multi_pinn(multi_count).a = p_m_a;
        end
    end
    
    % 해당 파일의 1-Step 오차 누적
    total_sq_err_1_s = total_sq_err_1_s + sum((preds_pinn_s - gt_s_compare).^2, 1);
    total_sq_err_1_p = total_sq_err_1_p + sum((preds_pinn_p - gt_p_compare).^2, 1);
    total_sq_err_1_a = total_sq_err_1_a + sum((preds_pinn_a - gt_a_compare).^2, 1);
    total_1_pts = total_1_pts + num_samples;
    
    % 해당 파일의 25-Step 오차 누적 및 최댓값 갱신
    for st = 1:multi_count
        g_idx   = multi_idx_list(st) + (1:step_horizon);       
        g_idx_p = multi_idx_list(st) + (2:step_horizon+1);     
        g_idx_a = multi_idx_list(st) + (0:step_horizon-1);     
        
        total_sq_err_s_step = total_sq_err_s_step + (multi_pinn(st).s - states_gt(g_idx, :)).^2;
        total_sq_err_p_step = total_sq_err_p_step + (multi_pinn(st).p - pos_gt(g_idx_p, :)).^2;
        total_sq_err_a_step = total_sq_err_a_step + (multi_pinn(st).a - accels_gt(g_idx_a, :)).^2;
        
        % --- 25-Step 전체 구간 대상 최대 절대 오차 갱신 ---
        max_err_25_s = max([max_err_25_s; abs(multi_pinn(st).s - states_gt(g_idx, :))], [], 1);
        max_err_25_p = max([max_err_25_p; abs(multi_pinn(st).p - pos_gt(g_idx_p, :))], [], 1);
        max_err_25_a = max([max_err_25_a; abs(multi_pinn(st).a - accels_gt(g_idx_a, :))], [], 1);
        
        % 글로벌 좌표 (X, Y) 유클리드 변위 계산
        pred_xy = multi_pinn(st).p(:, 1:2);
        gt_xy = pos_gt(g_idx_p, 1:2);
        dist_val = sqrt(sum((pred_xy - gt_xy).^2, 2)); 
        
        % 최대 FDE 갱신
        max_err_fde = max(max_err_fde, dist_val(end));
        
        % 스텝별 변위 및 전체 누적 저장
        total_disp_err_step = total_disp_err_step + dist_val; 
        total_ade_err = total_ade_err + sum(dist_val);  
        total_fde_err = total_fde_err + dist_val(end);  
        total_seq_count = total_seq_count + 1;   
    end
    
    fprintf('완료.\n');
end

%% 5. 결과 계산 및 시각화
if total_1_pts > 0
    rmse_1_s = sqrt(total_sq_err_1_s / double(total_1_pts));
    rmse_1_p = sqrt(total_sq_err_1_p / double(total_1_pts));
    rmse_1_a = sqrt(total_sq_err_1_a / double(total_1_pts));
end

if total_seq_count > 0
    % 스텝별 RMSE 배열
    rmse_s_step = sqrt(total_sq_err_s_step / total_seq_count); 
    rmse_p_step = sqrt(total_sq_err_p_step / total_seq_count); 
    rmse_a_step = sqrt(total_sq_err_a_step / total_seq_count); 
    
    % 각 물리량별 누적 오차 평균
    mean_rmse_s = mean(rmse_s_step, 1);
    mean_rmse_p = mean(rmse_p_step, 1);
    mean_rmse_a = mean(rmse_a_step, 1);
    
    % --- 스텝별 ADE & FDE 연산 ---
    fde_step_trend = total_disp_err_step / total_seq_count;
    ade_step_trend = cumsum(fde_step_trend) ./ (1:step_horizon)';
    
    avg_ade = total_ade_err / (total_seq_count * step_horizon);
    avg_fde = total_fde_err / total_seq_count;
    
    % --- 1. 텍스트 결과 출력 ---
    fprintf('\n====================================================================================\n');
    fprintf('                    [다중 파일 종합] Deep Dynamics Performance                     \n');
    fprintf('       테스트 데이터 수: %d개 파일 / 종합 평가 시퀀스: %d 개 \n', num_files, total_seq_count);
    fprintf('====================================================================================\n');
    fprintf(' [1-Step Point Prediction 오차 분석 (평균 RMSE / 최대 절대 오차)]\n');
    fprintf('  - States (vx, vy, r)  : Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f]\n', ...
        rmse_1_s(1), rmse_1_s(2), rmse_1_s(3), max_err_1_s(1), max_err_1_s(2), max_err_1_s(3));
    fprintf('  - Position (X, Y, psi): Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f] (2-Step)\n', ...
        rmse_1_p(1), rmse_1_p(2), rmse_1_p(3), max_err_1_p(1), max_err_1_p(2), max_err_1_p(3));
    fprintf('  - Accel (ax, ay)      : Avg[%.6f, %.6f]           / Max[%.6f, %.6f]\n', ...
        rmse_1_a(1), rmse_1_a(2), max_err_1_a(1), max_err_1_a(2));
    fprintf('------------------------------------------------------------------------------------\n');
    fprintf(' [25-Step Rollout 누적 오차 분석 (구간 평균 RMSE / 전 구간 최대 절대 오차)]\n');
    fprintf('  - States (vx, vy, r)  : Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f]\n', ...
        mean_rmse_s(1), mean_rmse_s(2), mean_rmse_s(3), max_err_25_s(1), max_err_25_s(2), max_err_25_s(3));
    fprintf('  - Position (X, Y, psi): Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f]\n', ...
        mean_rmse_p(1), mean_rmse_p(2), mean_rmse_p(3), max_err_25_p(1), max_err_25_p(2), max_err_25_p(3));
    fprintf('  - Accel (ax, ay)      : Avg[%.6f, %.6f]           / Max[%.6f, %.6f]\n', ...
        mean_rmse_a(1), mean_rmse_a(2), max_err_25_a(1), max_err_25_a(2));
    fprintf('  - Trajectory Error    : ADE = %.6f m, FDE = %.6f m (Max FDE = %.6f m)\n', ...
        avg_ade, avg_fde, max_err_fde);
    fprintf('====================================================================================\n');
    
    % --- 2. 그래프 시각화 ---
    figure('Name', 'Step-by-Step Prediction RMSE', 'Position', [100, 100, 1400, 800]);
    x_sa = 1:25;  % States, Accel X축
    x_p  = 2:26;  % Position X축 
    
    % States Plot
    subplot(3, 3, 1); plot(x_sa, rmse_s_step(:, 1), 'r-o', 'LineWidth', 1.5); title('vx RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 2); plot(x_sa, rmse_s_step(:, 2), 'g-o', 'LineWidth', 1.5); title('vy RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 3); plot(x_sa, rmse_s_step(:, 3), 'b-o', 'LineWidth', 1.5); title('Yaw Rate (r) RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    
    % Position Plot
    subplot(3, 3, 4); plot(x_p, rmse_p_step(:, 1), 'r-o', 'LineWidth', 1.5); title('X Position RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 5); plot(x_p, rmse_p_step(:, 2), 'g-o', 'LineWidth', 1.5); title('Y Position RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 6); plot(x_p, rmse_p_step(:, 3), 'b-o', 'LineWidth', 1.5); title('Heading (psi) RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    
    % Accel Plot
    subplot(3, 3, 7); plot(x_sa, rmse_a_step(:, 1), 'r-o', 'LineWidth', 1.5); title('ax RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 8); plot(x_sa, rmse_a_step(:, 2), 'g-o', 'LineWidth', 1.5); title('ay RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    
    % ADE / FDE Step-by-Step Plot
    subplot(3, 3, 9); 
    plot(x_p, fde_step_trend, 'm-o', 'LineWidth', 1.5, 'DisplayName', 'FDE (Step Error)'); hold on;
    plot(x_p, ade_step_trend, 'c-o', 'LineWidth', 1.5, 'DisplayName', 'ADE (Cumulative)');
    title('Displacement Error (ADE & FDE)'); 
    xlabel('Prediction Step'); ylabel('Error [m]'); 
    legend('Location', 'northwest'); grid on;
    
    % --- 3. 비교용 데이터 .mat 저장 (파일명 자동 생성) ---
    [~, model_name, ~] = fileparts(model_path);
    save_filename = sprintf('EvalResults_%s.mat', model_name);
    
    % 구조체 데이터 저장 (최댓값 정보 포함)
    eval_metrics.rmse_1_s       = rmse_1_s;
    eval_metrics.rmse_1_p       = rmse_1_p;
    eval_metrics.rmse_1_a       = rmse_1_a;
    eval_metrics.max_err_1_s    = max_err_1_s;
    eval_metrics.max_err_1_p    = max_err_1_p;
    eval_metrics.max_err_1_a    = max_err_1_a;
    
    eval_metrics.rmse_s_step    = rmse_s_step;
    eval_metrics.rmse_p_step    = rmse_p_step;
    eval_metrics.rmse_a_step    = rmse_a_step;
    eval_metrics.max_err_25_s   = max_err_25_s;
    eval_metrics.max_err_25_p   = max_err_25_p;
    eval_metrics.max_err_25_a   = max_err_25_a;
    
    eval_metrics.fde_step_trend = fde_step_trend;
    eval_metrics.ade_step_trend = ade_step_trend;
    eval_metrics.avg_ade        = avg_ade;
    eval_metrics.avg_fde        = avg_fde;
    eval_metrics.max_err_fde    = max_err_fde;
    eval_metrics.x_sa           = x_sa;
    eval_metrics.x_p            = x_p;
    
    save(save_filename, 'eval_metrics');
    fprintf('\n[Info] 최대 오차 분석 데이터가 구조체에 포함되어 저장되었습니다: %s\n', save_filename);
    
else
    fprintf('평가할 유효 시퀀스 데이터가 부족합니다.\n');
end

%% --- [함수부] ---
function loc_pos = global_to_local_matlab(pos_seq, ref_pos)
    dx = pos_seq(:, 1) - ref_pos(1); 
    dy = pos_seq(:, 2) - ref_pos(2); 
    dpsi = pos_seq(:, 3) - ref_pos(3);
    
    cos_p = cos(ref_pos(3)); sin_p = sin(ref_pos(3));
    loc_x = dx .* cos_p + dy .* sin_p; 
    loc_y = -dx .* sin_p + dy .* cos_p;
    
    loc_pos = [loc_x, loc_y, dpsi];
end

function [tf, tr, dr, Iz_pred, h_next] = get_pinn_params_stateful(input_data, md, p, tuning, h_prev)
    persistent W_ih b_ih W_hh b_hh W_mlp b_mlp W_out b_out
    persistent p_hidden_dim p_gru_layers p_fc_layers p_act_type
    persistent p_t_min p_t_max p_dr_min p_dr_max p_Iz_min p_Iz_max
    persistent Is_Initialized
    
    if isempty(Is_Initialized)
        p_hidden_dim = tuning.hidden_dim;
        p_gru_layers = tuning.num_gru_layers;
        p_fc_layers  = tuning.num_fc_layers;
        p_act_type   = tuning.act_type;
        
        p_t_min = p.t_min; p_t_max = p.t_max;
        p_dr_min = p.dr_min; p_dr_max = p.dr_max;
        p_Iz_min = p.Iz_min; p_Iz_max = p.Iz_max;
        
        W_ih = cell(p_gru_layers, 1); b_ih = cell(p_gru_layers, 1);
        W_hh = cell(p_gru_layers, 1); b_hh = cell(p_gru_layers, 1);
        
        for l = 1:p_gru_layers
            suffix = sprintf('_l%d', l-1);
            W_ih{l} = md.(['gru_weight_ih', suffix]);
            W_hh{l} = md.(['gru_weight_hh', suffix]);
            b_ih{l} = md.(['gru_bias_ih', suffix])(:);
            b_hh{l} = md.(['gru_bias_hh', suffix])(:);
        end
        
        W_mlp = cell(1 + p_fc_layers, 1);
        b_mlp = cell(1 + p_fc_layers, 1);
        
        W_mlp{1} = md.mlp_backbone_0_weight;
        b_mlp{1} = md.mlp_backbone_0_bias(:);
        
        for f = 1:p_fc_layers
            block_idx = 3*f; 
            W_mlp{1+f} = md.(sprintf('mlp_backbone_%d_weight', block_idx));
            b_mlp{1+f} = md.(sprintf('mlp_backbone_%d_bias', block_idx))(:);
        end
        
        W_out = md.output_layer_weight;
        b_out = md.output_layer_bias(:);
        
        Is_Initialized = true;
    end
    
    if isempty(h_prev)
        h = zeros(p_hidden_dim, p_gru_layers);
    else
        h = h_prev;
    end
    
    idx1 = p_hidden_dim; idx2 = 2 * p_hidden_dim; idx3 = 3 * p_hidden_dim;
    seq_len = size(input_data, 1);
    
    for t = 1:seq_len
        x_t = input_data(t, :)'; 
        for l = 1:p_gru_layers
            gi = W_ih{l} * x_t + b_ih{l}; 
            gh = W_hh{l} * h(:, l) + b_hh{l};
            
            r_gate = 1 ./ (1 + exp(-(gi(1:idx1) + gh(1:idx1))));
            z_gate = 1 ./ (1 + exp(-(gi(idx1+1:idx2) + gh(idx1+1:idx2))));
            n_gate = tanh(gi(idx2+1:idx3) + r_gate .* gh(idx2+1:idx3));
            
            h(:, l) = (1 - z_gate) .* n_gate + z_gate .* h(:, l);
            x_t = h(:, l); 
        end
    end
    h_next = h;
    
    x = h(:, end); 
    for i = 1:numel(W_mlp)
        x = W_mlp{i} * x + b_mlp{i};
        x = custom_activate(x, p_act_type);
    end
    
    out_sig = 1 ./ (1 + exp(-(W_out * x + b_out)));
    
    tf = p_t_min + (p_t_max - p_t_min) .* out_sig(1:6)';
    tr = p_t_min + (p_t_max - p_t_min) .* out_sig(7:12)';
    dr = p_dr_min + (p_dr_max - p_dr_min) .* out_sig(13:17)'; 
    Iz_pred = p_Iz_min + (p_Iz_max - p_Iz_min) * out_sig(18); 
end

function out = custom_activate(in, act_type)
    act_str = string(act_type); 
    if act_str == "tanh"
        out = tanh(in);
    elseif act_str == "gelu"
        out = 0.5 .* in .* (1 + tanh(sqrt(2/pi) .* (in + 0.044715 .* in.^3)));
    elseif act_str == "relu"
        out = max(0, in);
    else
        out = tanh(in);
    end
end

function [next_s, next_p, acc_body] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz, p)
    persistent p_m p_lf p_lr p_dt Is_Phys_Init
    
    if isempty(Is_Phys_Init)
        p_m = p.m; p_lf = p.lf; p_lr = p.lr; p_dt = p.dt;
        Is_Phys_Init = true;
    end
    
    vx = curr_s(1); vy = curr_s(2); r = curr_s(3); 
    X = curr_p(1); Y = curr_p(2); psi = curr_p(3); 
    T = curr_c(1); brake = curr_c(2); delta = curr_c(3);
    
    Frx = (dr(1) - dr(2)*vx)*T - dr(5) * brake - dr(3) - dr(4)*vx^2;
    af = delta - atan2(r*p_lf + vy, vx + 1e-6) + tf(5); 
    ar = atan2(r*p_lr - vy, vx + 1e-6) + tr(5);
    
    Ffy = tf(6) + tf(3)*sin(tf(2)*atan(tf(1)*af - tf(4)*(tf(1)*af - atan(tf(1)*af))));
    Fry = tr(6) + tr(3)*sin(tr(2)*atan(tr(1)*ar - tr(4)*(tr(1)*ar - atan(tr(1)*ar))));
    
    vx_dot = (- Ffy*sin(delta) + Frx)/p_m + vy*r; 
    vy_dot = (  Ffy*cos(delta) + Fry)/p_m - vx*r;
    r_dot = (Ffy*cos(delta)*p_lf - Fry*p_lr) / Iz;
    
    next_s = [vx + vx_dot*p_dt, vy + vy_dot*p_dt, r + r_dot*p_dt];
    next_p = [X + (vx*cos(psi) - vy*sin(psi))*p_dt, Y + (vx*sin(psi) + vy*cos(psi))*p_dt, psi + r*p_dt];
    acc_body = [vx_dot - vy*r, vy_dot + vx*r];
end