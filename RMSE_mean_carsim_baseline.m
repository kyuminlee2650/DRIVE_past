%% =========================================================================
% Deep Dynamics Baseline - 전체 통합 검증 스크립트 (다중 CSV 배치 평가)
% 수정 사항: 
% 1. 시점 동기화: 가속도(0~24스텝), 상태(1~25스텝), 위치(2~26스텝) 오차 타깃 분리
% 2. 연산 최적화: get_pinn_params_stateful 및 physics_step 내 Persistent 변수 적용
% 3. 출력 형식 변경: Avg / Max 한 줄 정렬 및 Max FDE 추적 추가
% =========================================================================
clear vars; close all; clear functions;

%% 1. 설정 및 초기화
model_path = "C:\Users\mumu2\Downloads\baseline_tuning_trial_1_best_model.mat"; 
model_data = load(model_path);
csv_folder = "C:\workspace\carsim_processed_data_test_25hz";
csv_files = dir(fullfile(csv_folder, '*.csv'));
num_files = length(csv_files);

tuning.seq_len        = model_data.opt_seq_len;       
tuning.hidden_dim     = model_data.opt_hidden_dim;    
tuning.num_gru_layers = model_data.opt_num_layers;    
tuning.num_fc_layers  = model_data.opt_num_fc_layers; 
tuning.act_type       = model_data.opt_act_func;      

if num_files == 0
    err_msg = sprintf('지정된 폴더에 CSV 파일이 존재하지 않습니다: %s', csv_folder);
    disp(err_msg);
    return;
end

% 차량 파라미터 및 모델 정보 바인딩
params.dt = 0.04; 
params.m = 1270;
params.lf = 1.015; params.lr = 1.895;
params.seq_len = tuning.seq_len; 
params.t_min = model_data.t_min; params.t_max = model_data.t_max;
params.dr_min = model_data.dr_min; params.dr_max = model_data.dr_max;
params.Iz_min = model_data.Iz_min; params.Iz_max = model_data.Iz_max;

% 정규화 데이터
mean_vec = reshape(model_data.in_mean, 1, []);
std_vec  = reshape(model_data.in_std, 1, []);

% 글로벌 누적 오차 변수 초기화 (1-step 예측)
total_sq_err_1_s = zeros(1, 3); total_sq_err_1_p = zeros(1, 3); total_sq_err_1_a = zeros(1, 2);
max_err_1_s = zeros(1, 3); max_err_1_p = zeros(1, 3); max_err_1_a = zeros(1, 2);
total_1_pts = 0;

% 스텝별 오차 누적용 변수 (25스텝 기준)
step_horizon = 25;
total_sq_err_s_step = zeros(step_horizon, 3); max_err_s_step = zeros(step_horizon, 3);
total_sq_err_p_step = zeros(step_horizon, 3); max_err_p_step = zeros(step_horizon, 3);
total_sq_err_a_step = zeros(step_horizon, 2); max_err_a_step = zeros(step_horizon, 2);

% 스텝별 Displacement 오차 누적용 변수
total_disp_err_step = zeros(step_horizon, 1);
total_ade_err = 0;
total_fde_err = 0;
max_fde = 0; % [추가] 최대 FDE 추적 변수
total_seq_count = 0;

fprintf('총 %d개의 파일을 찾았습니다. 평가를 시작합니다...\n\n', num_files);

%% 2. 다중 파일 순회 루프
for f_idx = 1:num_files
    data_path = fullfile(csv_folder, csv_files(f_idx).name);
    fprintf('처리 중 [%d/%d]: %s ... ', f_idx, num_files, csv_files(f_idx).name);
    data = readmatrix(data_path, 'NumHeaderLines', 1);
    
    % 데이터 파싱
    time      = data(:, 1);
    pos_gt    = data(:, 2:4);      
    states_gt = data(:, 5:7);      
    ctrls     = data(:, 8:10);     
    accels_gt = data(:, 11:12);    
    
    % 7차원 입력 피처 구성
    T_raw = ctrls(:, 1);
    d_raw = ctrls(:, 3);
    dT = [0; diff(T_raw)];
    d_d = [0; diff(d_raw)];
    X_inputs = [states_gt, T_raw, d_raw, dT, d_d];
    
    num_samples = size(data, 1) - params.seq_len - 2;
    if num_samples < 1
        fprintf('데이터가 짧아 건너뜁니다.\n');
        continue; 
    end
    
    %% 3. 샘플 루프 및 시뮬레이션
    for i = 1:num_samples
        idx = i + params.seq_len - 1;
        curr_s = states_gt(idx, :); curr_p = pos_gt(idx, :); 
        curr_c = ctrls(idx, :);
        next_c = ctrls(idx+1, :); 
        
        % 입력 시퀀스 및 파라미터 추론
        seq_input = X_inputs(i:idx, :); 
        input_norm = (seq_input - mean_vec) ./ (std_vec + 1e-6);
        [tf, tr, dr, Iz_pred, ~] = get_pinn_params_stateful(input_norm, model_data, params, tuning);
        
        % 1-Step 예측 기록
        [p_s_t1, p_p_t1, p_a_t] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz_pred, params);
        [~, p_p_t2, ~] = physics_step(p_s_t1, p_p_t1, next_c, tf, tr, dr, Iz_pred, params);
        
        err_1_s = abs(p_s_t1 - states_gt(idx+1, :));
        err_1_p = abs(p_p_t2 - pos_gt(idx+2, :));
        err_1_a = abs(p_a_t - accels_gt(idx, :));

        total_sq_err_1_s = total_sq_err_1_s + err_1_s.^2;
        total_sq_err_1_p = total_sq_err_1_p + err_1_p.^2;
        total_sq_err_1_a = total_sq_err_1_a + err_1_a.^2;
        
        max_err_1_s = max(max_err_1_s, err_1_s);
        max_err_1_p = max(max_err_1_p, err_1_p);
        max_err_1_a = max(max_err_1_a, err_1_a);

        total_1_pts = total_1_pts + 1;
        
        % 25-Step 다중 시퀀스 롤아웃 및 스텝 매칭
        if mod(i, 25) == 0 && time(idx) >= 1.0 && time(idx) <= time(end-100)
            ps_tmp = curr_s; pp_tmp = curr_p;
            
            local_sq_s = zeros(step_horizon, 3); local_abs_s = zeros(step_horizon, 3);
            local_sq_p = zeros(step_horizon, 3); local_abs_p = zeros(step_horizon, 3);
            local_sq_a = zeros(step_horizon, 2); local_abs_a = zeros(step_horizon, 2);
            local_dist_err = zeros(step_horizon + 1, 1);
            
            seq_completed = true;
            
            for h_step = 1:(step_horizon + 1) 
                f_step_idx = idx + h_step - 1;
                
                if f_step_idx + 1 > size(ctrls, 1) || f_step_idx + 1 > size(pos_gt, 1)
                    seq_completed = false;
                    break; 
                end
                
                f_ctrl = ctrls(f_step_idx, :);
                [ps_tmp, pp_tmp, pa_tmp] = physics_step(ps_tmp, pp_tmp, f_ctrl, tf, tr, dr, Iz_pred, params);
                
                if h_step <= step_horizon
                    gt_s = states_gt(f_step_idx+1, :);
                    gt_a = accels_gt(f_step_idx, :); 
                    err_s = abs(ps_tmp - gt_s);
                    err_a = abs(pa_tmp - gt_a);
                    local_sq_s(h_step, :) = err_s.^2;
                    local_sq_a(h_step, :) = err_a.^2;
                    local_abs_s(h_step, :) = err_s;
                    local_abs_a(h_step, :) = err_a;
                end
                
                if h_step >= 2
                    gt_p = pos_gt(f_step_idx+1, :);
                    err_p = abs(pp_tmp - gt_p);
                    local_sq_p(h_step-1, :) = err_p.^2;
                    local_abs_p(h_step-1, :) = err_p; 
                    
                    pred_xy = pp_tmp(1:2);
                    gt_xy = gt_p(1:2);
                    local_dist_err(h_step) = sqrt(sum((pred_xy - gt_xy).^2));
                end
            end
            
            if seq_completed
                total_sq_err_s_step = total_sq_err_s_step + local_sq_s;
                total_sq_err_p_step = total_sq_err_p_step + local_sq_p;
                total_sq_err_a_step = total_sq_err_a_step + local_sq_a;
                
                max_err_s_step = max(max_err_s_step, local_abs_s);
                max_err_p_step = max(max_err_p_step, local_abs_p);
                max_err_a_step = max(max_err_a_step, local_abs_a);

                total_disp_err_step = total_disp_err_step + local_dist_err(2:end);
                
                total_ade_err = total_ade_err + sum(local_dist_err(2:end));
                
                curr_fde = local_dist_err(end);
                total_fde_err = total_fde_err + curr_fde;
                max_fde = max(max_fde, curr_fde); % [추가] 최대 FDE 누적
                
                total_seq_count = total_seq_count + 1;
            end
        end
    end
    fprintf('완료.\n');
end

%% 4. 결과 계산 및 시각화 출력
if total_1_pts > 0
    rmse_1_s = sqrt(total_sq_err_1_s / total_1_pts);
    rmse_1_p = sqrt(total_sq_err_1_p / total_1_pts);
    rmse_1_a = sqrt(total_sq_err_1_a / total_1_pts);
end

if total_seq_count > 0
    rmse_s_step = sqrt(total_sq_err_s_step / total_seq_count); 
    rmse_p_step = sqrt(total_sq_err_p_step / total_seq_count); 
    rmse_a_step = sqrt(total_sq_err_a_step / total_seq_count); 
    
    mean_rmse_s = mean(rmse_s_step, 1);
    mean_rmse_p = mean(rmse_p_step, 1);
    mean_rmse_a = mean(rmse_a_step, 1);
    
    overall_max_25_s = max(max_err_s_step, [], 1);
    overall_max_25_p = max(max_err_p_step, [], 1);
    overall_max_25_a = max(max_err_a_step, [], 1);

    fde_step_trend = total_disp_err_step / total_seq_count;
    ade_step_trend = cumsum(fde_step_trend) ./ (1:step_horizon)';
    
    avg_ade = total_ade_err / (total_seq_count * step_horizon);
    avg_fde = total_fde_err / total_seq_count;
    
    fprintf('\n==========================================================\n');
    fprintf('     [Baseline 통합 평가] 종합 결과 요약 (%d Sequences)   \n', total_seq_count);
    fprintf('==========================================================\n');
    
    % [1-Step Point Prediction 오차 분석]
    fprintf('\n[1-Step Point Prediction 오차 분석 (평균 RMSE / 최대 절대 오차)]\n');
    fprintf('States (vx, vy, r)  : Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f]\n', ...
        rmse_1_s(1), rmse_1_s(2), rmse_1_s(3), max_err_1_s(1), max_err_1_s(2), max_err_1_s(3));
    fprintf('Position (X, Y, psi): Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f] (2-Step)\n', ...
        rmse_1_p(1), rmse_1_p(2), rmse_1_p(3), max_err_1_p(1), max_err_1_p(2), max_err_1_p(3));
    fprintf('Accel (ax, ay)      : Avg[%.6f, %.6f] / Max[%.6f, %.6f]\n', ...
        rmse_1_a(1), rmse_1_a(2), max_err_1_a(1), max_err_1_a(2));
        
    % [25-Step Rollout 누적 오차 분석]
    fprintf('\n[25-Step Rollout 누적 오차 분석 (구간 평균 RMSE / 전 구간 최대 절대 오차)]\n');
    fprintf('States (vx, vy, r)  : Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f]\n', ...
        mean_rmse_s(1), mean_rmse_s(2), mean_rmse_s(3), overall_max_25_s(1), overall_max_25_s(2), overall_max_25_s(3));
    fprintf('Position (X, Y, psi): Avg[%.6f, %.6f, %.6f] / Max[%.6f, %.6f, %.6f]\n', ...
        mean_rmse_p(1), mean_rmse_p(2), mean_rmse_p(3), overall_max_25_p(1), overall_max_25_p(2), overall_max_25_p(3));
    fprintf('Accel (ax, ay)      : Avg[%.6f, %.6f] / Max[%.6f, %.6f]\n', ...
        mean_rmse_a(1), mean_rmse_a(2), overall_max_25_a(1), overall_max_25_a(2));
    fprintf('Trajectory Error    : ADE = %.6f m, FDE = %.6f m (Max FDE = %.6f m)\n', ...
        avg_ade, avg_fde, max_fde);
    fprintf('==========================================================\n');
    
    % 그래프 시각화
    figure('Name', 'Step-by-Step Prediction RMSE', 'Position', [100, 100, 1400, 800]);
    x_s  = 1:25;  x_p  = 2:26;  x_a  = 0:24;  
    
    subplot(3, 3, 1); plot(x_s, rmse_s_step(:, 1), 'r-o', 'LineWidth', 1.5); title('vx RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 2); plot(x_s, rmse_s_step(:, 2), 'g-o', 'LineWidth', 1.5); title('vy RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 3); plot(x_s, rmse_s_step(:, 3), 'b-o', 'LineWidth', 1.5); title('Yaw Rate (r) RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    
    subplot(3, 3, 4); plot(x_p, rmse_p_step(:, 1), 'r-o', 'LineWidth', 1.5); title('X Position RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 5); plot(x_p, rmse_p_step(:, 2), 'g-o', 'LineWidth', 1.5); title('Y Position RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    subplot(3, 3, 6); plot(x_p, rmse_p_step(:, 3), 'b-o', 'LineWidth', 1.5); title('Heading (psi) RMSE'); xlabel('Prediction Step'); ylabel('RMSE'); grid on;
    
    subplot(3, 3, 7); plot(x_a, rmse_a_step(:, 1), 'r-o', 'LineWidth', 1.5); title('ax RMSE'); xlabel('Prediction Step (t~t+24)'); ylabel('RMSE'); grid on;
    subplot(3, 3, 8); plot(x_a, rmse_a_step(:, 2), 'g-o', 'LineWidth', 1.5); title('ay RMSE'); xlabel('Prediction Step (t~t+24)'); ylabel('RMSE'); grid on;
    
    subplot(3, 3, 9); 
    plot(x_p, fde_step_trend, 'm-o', 'LineWidth', 1.5, 'DisplayName', 'FDE (Step Error)'); hold on;
    plot(x_p, ade_step_trend, 'c-o', 'LineWidth', 1.5, 'DisplayName', 'ADE (Cumulative)');
    title('Displacement Error (ADE & FDE)'); 
    xlabel('Prediction Step (t+2 ~ t+26)'); ylabel('Error [m]'); 
    legend('Location', 'northwest'); grid on;
    
    % --- 5. 비교용 데이터 .mat 저장 ---
    [~, model_name, ~] = fileparts(model_path);
    save_filename = sprintf('EvalResults_Baseline_%s.mat', model_name);
    
    eval_metrics.rmse_1_s       = rmse_1_s;
    eval_metrics.rmse_1_p       = rmse_1_p;
    eval_metrics.rmse_1_a       = rmse_1_a;
    eval_metrics.rmse_s_step    = rmse_s_step;
    eval_metrics.rmse_p_step    = rmse_p_step;
    eval_metrics.rmse_a_step    = rmse_a_step;
    eval_metrics.fde_step_trend = fde_step_trend;
    eval_metrics.ade_step_trend = ade_step_trend;
    
    eval_metrics.max_err_1_s    = max_err_1_s;
    eval_metrics.max_err_1_p    = max_err_1_p;
    eval_metrics.max_err_1_a    = max_err_1_a;
    eval_metrics.max_err_s_step = max_err_s_step;
    eval_metrics.max_err_p_step = max_err_p_step;
    eval_metrics.max_err_a_step = max_err_a_step;
    
    eval_metrics.avg_ade        = avg_ade;
    eval_metrics.avg_fde        = avg_fde;
    eval_metrics.max_fde        = max_fde; % [추가] 구조체에도 최대 FDE 저장
    eval_metrics.x_s            = x_s;
    eval_metrics.x_p            = x_p;
    eval_metrics.x_a            = x_a;
    
    save(save_filename, 'eval_metrics');
    fprintf('\n[Info] 비교용 평가 데이터가 저장되었습니다: %s\n', save_filename);
    
else
    fprintf('평가할 유효 시퀀스 데이터가 부족합니다.\n');
end

%% --- [함수부 최적화] Persistent 캐싱 적용 ---
function [tf, tr, dr, Iz_pred, h_next] = get_pinn_params_stateful(input_data, md, p, tuning)
    persistent Is_Initialized p_h_dim p_num_gru p_num_fc p_act_type
    persistent W_ih b_ih W_hh b_hh W_mlp b_mlp W_out b_out
    persistent p_t_min p_t_max p_dr_min p_dr_max p_Iz_min p_Iz_max
    
    if isempty(Is_Initialized)
        p_h_dim = tuning.hidden_dim;
        p_num_gru = tuning.num_gru_layers;
        p_num_fc = tuning.num_fc_layers;
        p_act_type = string(tuning.act_type); 
        
        p_t_min = p.t_min; p_t_max = p.t_max;
        p_dr_min = p.dr_min; p_dr_max = p.dr_max;
        p_Iz_min = p.Iz_min; p_Iz_max = p.Iz_max;
        
        W_ih = cell(p_num_gru, 1); W_hh = cell(p_num_gru, 1);
        b_ih = cell(p_num_gru, 1); b_hh = cell(p_num_gru, 1);
        for l = 1:p_num_gru
            layer_idx = l - 1;
            W_ih{l} = md.(sprintf('gru_weight_ih_l%d', layer_idx));
            W_hh{l} = md.(sprintf('gru_weight_hh_l%d', layer_idx));
            b_ih{l} = md.(sprintf('gru_bias_ih_l%d', layer_idx))(:);
            b_hh{l} = md.(sprintf('gru_bias_hh_l%d', layer_idx))(:);
        end
        
        W_mlp = cell(1 + p_num_fc, 1); b_mlp = cell(1 + p_num_fc, 1);
        W_mlp{1} = md.mlp_backbone_0_weight;
        b_mlp{1} = md.mlp_backbone_0_bias(:);
        
        for f = 1:p_num_fc
            block_idx = 3*f;
            W_mlp{1+f} = md.(sprintf('mlp_backbone_%d_weight', block_idx));
            b_mlp{1+f} = md.(sprintf('mlp_backbone_%d_bias', block_idx))(:);
        end
        
        W_out = md.output_layer_weight;
        b_out = md.output_layer_bias(:);
        
        Is_Initialized = true;
    end
    
    idx1 = p_h_dim; idx2 = 2 * p_h_dim; idx3 = 3 * p_h_dim;
    h_layers = zeros(p_h_dim, p_num_gru);
    
    for t = 1:size(input_data, 1)
        layer_input = input_data(t, :)';
        for l = 1:p_num_gru
            h_curr = h_layers(:, l);
            gi = W_ih{l} * layer_input + b_ih{l};
            gh = W_hh{l} * h_curr + b_hh{l};
            
            r_gate = 1 ./ (1 + exp(-(gi(1:idx1) + gh(1:idx1))));
            z_gate = 1 ./ (1 + exp(-(gi(idx1+1:idx2) + gh(idx1+1:idx2))));
            n_gate = tanh(gi(idx2+1:idx3) + r_gate .* gh(idx2+1:idx3));
            
            h_curr = (1 - z_gate) .* n_gate + z_gate .* h_curr;
            h_layers(:, l) = h_curr;
            layer_input = h_curr;
        end
    end
    
    h_next = h_layers(:, p_num_gru);
    x_feat = h_next;
    
    for i = 1:numel(W_mlp)
        x_feat = W_mlp{i} * x_feat + b_mlp{i};
        x_feat = custom_activate(x_feat, p_act_type);
    end
    
    out_sig = 1 ./ (1 + exp(-(W_out * x_feat + b_out)));
    
    tf = p_t_min + (p_t_max - p_t_min) .* out_sig(1:6)';
    tr = p_t_min + (p_t_max - p_t_min) .* out_sig(7:12)';
    dr = p_dr_min + (p_dr_max - p_dr_min) .* out_sig(13:16)'; 
    Iz_pred = p_Iz_min + (p_Iz_max - p_Iz_min) * out_sig(17);
end

function [next_s, next_p, acc_body] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz, p)
    persistent p_m p_lf p_lr p_dt Is_Phys_Init
    if isempty(Is_Phys_Init)
        p_m = p.m; p_lf = p.lf; p_lr = p.lr; p_dt = p.dt;
        Is_Phys_Init = true;
    end
    
    vx = curr_s(1); vy = curr_s(2); r = curr_s(3); X = curr_p(1); Y = curr_p(2); psi = curr_p(3); 
    T = curr_c(1); delta = curr_c(3);
    
    Frx = (dr(1) - dr(2)*vx)*T - dr(3) - dr(4)*(vx^2);
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