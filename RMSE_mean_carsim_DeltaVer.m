%% =========================================================================
% Deep Dynamics Model (12-Dim Input) - 다중 CSV 배치 평가 스크립트
% 수정 사항: 12입력 구조 확장 및 Sequential MLP 백본 레이어 매핑 적용
% =========================================================================
clear vars;      
clear functions; % persistent 변수 초기화를 위해 필수 호출
clear all; close all;

%% 0. ⚙️ [하이퍼파라미터 튜닝 선언부] - 모델 구조 제어
tuning.seq_len        = 16;     % 시퀀스 길이
tuning.hidden_dim     = 128;    % 은닉층(Hidden) 차원
tuning.num_gru_layers = 2;      % GRU 레이어 개수
tuning.num_fc_layers  = 3;      % MLP 백본 내 반복 추가 횟수 (PyTorch 모델의 num_fc_layers)
tuning.act_type       = 3;      % 활성화 함수 (1: 'tanh', 2: 'gelu', 3: 'relu')

%% 1. 모델 경로 및 테스트 데이터 폴더 설정
model_path = "C:\workspace\best\DRIVEDeltaVer_real_best.mat";
csv_folder = "C:\workspace\test_csv";

fprintf('모델 로드 중: %s\n', model_path);
model_data = load(model_path);
csv_files = dir(fullfile(csv_folder, '*.csv'));
num_files = length(csv_files);

if num_files == 0
    error('지정된 폴더에 CSV 파일이 존재하지 않습니다: %s', csv_folder);
end
fprintf('총 %d개의 테스트 CSV 파일을 발견했습니다. 평가를 시작합니다...\n\n', num_files);

%% 2. 차량 파라미터 기본 설정
params.dt = 0.04; 
params.m = 1014;
params.lf = 1.2; 
params.lr = 1.29;
params.seq_len = tuning.seq_len; % 튜닝 파라미터 바인딩

% 네트워크 출력 Boundary 파라미터
params.t_min = model_data.t_min;
params.t_max = model_data.t_max;
params.dr_min = model_data.dr_min;
params.dr_max = model_data.dr_max;
params.Iz_min = model_data.Iz_min;
params.Iz_max = model_data.Iz_max;

%% 3. 글로벌 누적 오차 변수 초기화 (전체 파일 대상)
total_sq_err_1_s = zeros(1, 3); total_sq_err_1_p = zeros(1, 3); total_sq_err_1_a = zeros(1, 2);
total_1_pts = 0;

total_sq_err_25_s = zeros(1, 3); total_sq_err_25_p = zeros(1, 3); total_sq_err_25_a = zeros(1, 2);
total_25_pts = 0;

total_ade_err = 0;
total_fde_err = 0;
total_25_sequences = 0;
horizon = 25; 

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
    pos_gt    = data(:, 2:4);         % X, Y, psi
    states_gt = data(:, 5:7);         % vx, vy, r
    ctrls     = data(:, 8:10);        % Throttle, Brake, delta
    accels_gt = data(:, 11:12);       % ax, ay
    
    % 로컬 버퍼 초기화
    preds_pinn_s = zeros(num_samples, 3); 
    preds_pinn_p = zeros(num_samples, 3); 
    preds_pinn_a = zeros(num_samples, 2);
    
    gt_s_compare = zeros(num_samples, 3); 
    gt_p_compare = zeros(num_samples, 3); 
    gt_a_compare = zeros(num_samples, 2); 
    
    multi_count = 0; 
    multi_idx_list = [];
    multi_pinn = struct('s', {}, 'p', {}, 'a', {}, 't', {}); 
    
    % 1-Step 및 25-Step 평가
    for i = 1:num_samples
        idx = i + params.seq_len - 1; % 현재 시점 t
        
        curr_t = time(idx);
        curr_s = states_gt(idx, :); 
        curr_p = pos_gt(idx, :); 
        curr_c = ctrls(idx, :);
        next_c = ctrls(idx+1, :); 
        
        % --- 12차원 입력 특성 시퀀스 구성 ---
        seq_pos   = pos_gt(i:idx, :);
        seq_states = states_gt(i:idx, :);
        seq_ctrls  = ctrls(i:idx, :);
        seq_accels = data(i:idx, 11:12); 
        
        % 12번째 확장 입력 feature 처리 (13번째 열이 없으면 0 패딩)
        if size(data, 2) >= 13
            seq_extra = data(i:idx, 13);
        else
            seq_extra = zeros(size(seq_accels, 1), 1);
        end
        
        mid_idx = floor(params.seq_len / 2) + 1;
        ref_pos = seq_pos(mid_idx, :);
        
        loc_pos_seq = global_to_local_matlab(seq_pos, ref_pos);
        
        % 12차원 텐서 조합 및 정규화
        in_unnorm = [loc_pos_seq, seq_states, seq_ctrls, seq_accels, seq_extra]; 
        input_seq = (in_unnorm - model_data.in_mean) ./ (model_data.in_std + 1e-6);
        
        % --- 1. 1-Step 예측 ---
        [tf, tr, dr, Iz_pred, ~] = get_pinn_params_stateful(input_seq, model_data, params, tuning, []);
        
        % 물리 엔진 스텝 구동
        [p_s_t1, p_p_t1, p_a_t] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz_pred, params);
        [~, p_p_t2, ~] = physics_step(p_s_t1, p_p_t1, next_c, tf, tr, dr, Iz_pred, params);
        
        preds_pinn_s(i, :) = p_s_t1; 
        preds_pinn_p(i, :) = p_p_t2; 
        preds_pinn_a(i, :) = p_a_t;
        
        % GT 매핑 저장
        gt_s_compare(i, :) = states_gt(idx+1, :);
        gt_p_compare(i, :) = pos_gt(idx+2, :);
        gt_a_compare(i, :) = accels_gt(idx, :);
        
        % --- 2. 25-Step Recursive 예측 ---
        if mod(i, 25) == 0 && curr_t >= 1.0 && curr_t <= time(end-100)
            multi_count = multi_count + 1;
            multi_idx_list(multi_count) = idx;
            
            ps_tmp = curr_s; pp_tmp = curr_p; 
            p_m_s = zeros(horizon, 3); p_m_p = zeros(horizon, 3); p_m_a = zeros(horizon, 2);
            
            for h_step = 1:horizon
                f_idx_h = idx + h_step - 1; 
                if f_idx_h+1 > size(ctrls, 1), break; end
                f_ctrl = ctrls(f_idx_h, :);
                
                [ps_tmp, pp_tmp, pa_tmp] = physics_step(ps_tmp, pp_tmp, f_ctrl, tf, tr, dr, Iz_pred, params);
                
                p_m_s(h_step,:) = ps_tmp;
                p_m_p(h_step,:) = pp_tmp; 
                p_m_a(h_step,:) = pa_tmp;
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
    
    % 해당 파일의 25-Step 오차 누적 및 ADE/FDE 계산
    for st = 1:multi_count
        g_idx = multi_idx_list(st) + (1:horizon);
        if g_idx(end) > size(data,1), break; end
        
        total_sq_err_25_s = total_sq_err_25_s + sum((multi_pinn(st).s - states_gt(g_idx,:)).^2, 1);
        total_sq_err_25_p = total_sq_err_25_p + sum((multi_pinn(st).p - pos_gt(g_idx,:)).^2, 1);
        total_sq_err_25_a = total_sq_err_25_a + sum((multi_pinn(st).a - accels_gt(g_idx,:)).^2, 1);
        total_25_pts = total_25_pts + horizon;
        
        % 글로벌 좌표계 변위 유클리드 오차 계산
        pred_xy = multi_pinn(st).p(:, 1:2);
        gt_xy = pos_gt(g_idx, 1:2);
        dist_err = sqrt(sum((pred_xy - gt_xy).^2, 2)); 
        
        total_ade_err = total_ade_err + sum(dist_err);  
        total_fde_err = total_fde_err + dist_err(end);  
        total_25_sequences = total_25_sequences + 1;   
    end
    
    fprintf('완료.\n');
end

%% 5. 전체 평균 RMSE, ADE, FDE 종합 계산
rmse_1_s_p = sqrt(total_sq_err_1_s / total_1_pts);
rmse_1_p_p = sqrt(total_sq_err_1_p / total_1_pts);
rmse_1_a_p = sqrt(total_sq_err_1_a / total_1_pts);

if total_25_pts > 0
    rmse_25_s_p = sqrt(total_sq_err_25_s / total_25_pts); 
    rmse_25_p_p = sqrt(total_sq_err_25_p / total_25_pts); 
    rmse_25_a_p = sqrt(total_sq_err_25_a / total_25_pts); 
    
    avg_ade = total_ade_err / (total_25_sequences * horizon);
    avg_fde = total_fde_err / total_25_sequences;
else
    rmse_25_s_p = [NaN, NaN, NaN]; rmse_25_p_p = [NaN, NaN, NaN]; rmse_25_a_p = [NaN, NaN];
    avg_ade = NaN; avg_fde = NaN;
end

%% 6. 최종 통합 결과 출력 
fprintf('\n==========================================================\n');
fprintf('     [다중 파일 종합] Deep Dynamics (12-In PINN) 성능    \n');
fprintf('     테스트 데이터 수: %d개 파일 / 총 샘플 수: %d 개       \n', num_files, total_1_pts);
fprintf('==========================================================\n');
fprintf(' [1-Step Prediction Average RMSE]\n');
fprintf('  - States (vx, vy, r)  : [%.6f, %.6f, %.6f]\n', rmse_1_s_p(1), rmse_1_s_p(2), rmse_1_s_p(3));
fprintf('  - Position (X, Y, psi): [%.6f, %.6f, %.6f]\n', rmse_1_p_p(1), rmse_1_p_p(2), rmse_1_p_p(3));
fprintf('  - Accel (ax, ay)      : [%.6f, %.6f]\n', rmse_1_a_p(1), rmse_1_a_p(2));
fprintf('----------------------------------------------------------\n');
fprintf(' [25-Step Recursive Average Performance]\n');
fprintf('  - States (vx, vy, r)  : [%.6f, %.6f, %.6f]\n', rmse_25_s_p(1), rmse_25_s_p(2), rmse_25_s_p(3));
fprintf('  - Position (X, Y, psi): [%.6f, %.6f, %.6f]\n', rmse_25_p_p(1), rmse_25_p_p(2), rmse_25_p_p(3));
fprintf('  - Accel (ax, ay)      : [%.6f, %.6f]\n', rmse_25_a_p(1), rmse_25_a_p(2));
fprintf('  - Trajectory Error    : ADE = %.6f m, FDE = %.6f m\n', avg_ade, avg_fde);
fprintf('==========================================================\n');

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
    hidden_dim = tuning.hidden_dim;
    gru_layers = tuning.num_gru_layers;
    fc_layers  = tuning.num_fc_layers;
    act_type   = tuning.act_type;
    
    persistent W_ih b_ih W_hh b_hh W_mlp b_mlp W_out b_out Is_Initialized
    
    if isempty(Is_Initialized)
        % --- 1. GRU 다중 레이어 가중치 로드 ---
        W_ih = cell(gru_layers, 1); b_ih = cell(gru_layers, 1);
        W_hh = cell(gru_layers, 1); b_hh = cell(gru_layers, 1);
        
        for l = 1:gru_layers
            suffix = sprintf('_l%d', l-1);
            W_ih{l} = md.(['gru_weight_ih', suffix]);
            W_hh{l} = md.(['gru_weight_hh', suffix]);
            b_ih{l} = md.(['gru_bias_ih', suffix])(:); 
            b_hh{l} = md.(['gru_bias_hh', suffix])(:);
        end
        
        % --- 2. nn.Sequential 순차 레이어 로드 구조 반영 ---
        W_mlp = cell(1 + fc_layers, 1);
        b_mlp = cell(1 + fc_layers, 1);
        
        % 초기 레이어 (layers[0])
        W_mlp{1} = md.mlp_backbone_0_weight;
        b_mlp{1} = md.mlp_backbone_0_bias(:);
        
        % PyTorch nn.Sequential 선형 레이어 인덱싱 수식 적용 (2, 5, 8, 11...)
        for f = 1:fc_layers
            layer_idx = 2 + 3 * (f - 1); 
            W_mlp{1+f} = md.(sprintf('mlp_backbone_%d_weight', layer_idx));
            b_mlp{1+f} = md.(sprintf('mlp_backbone_%d_bias', layer_idx))(:);
        end
        
        % 출력 레이어
        W_out = md.output_layer_weight;
        b_out = md.output_layer_bias(:);
        
        Is_Initialized = true;
    end
    
    if isempty(h_prev)
        h = zeros(hidden_dim, gru_layers);
    else
        h = h_prev;
    end
    
    % --- GRU Forward Pass ---
    idx1 = hidden_dim; idx2 = 2 * hidden_dim; idx3 = 3 * hidden_dim;
    seq_len = size(input_data, 1);
    
    for t = 1:seq_len
        x_t = input_data(t, :)'; 
        for l = 1:gru_layers
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
    
    % --- 순차 MLP 백본 연산 ---
    x = h(:, end); 
    for i = 1:numel(W_mlp)
        x = W_mlp{i} * x + b_mlp{i};
        x = custom_activate(x, act_type);
    end
    
    % --- Sigmoid 및 18차원 파라미터 분할 매핑 ---
    out_sig = 1 ./ (1 + exp(-(W_out * x + b_out)));
    
    tf = p.t_min + (p.t_max - p.t_min) .* out_sig(1:6)';
    tr = p.t_min + (p.t_max - p.t_min) .* out_sig(7:12)';
    dr = p.dr_min + (p.dr_max - p.dr_min) .* out_sig(13:17)'; 
    Iz_pred = p.Iz_min + (p.Iz_max - p.Iz_min) * out_sig(18); 
end

function out = custom_activate(in, act_type)
    if act_type == 1
        out = tanh(in);
    elseif act_type == 2
        out = 0.5 .* in .* (1 + tanh(sqrt(2/pi) .* (in + 0.044715 .* in.^3)));
    elseif act_type == 3
        out = max(0, in);
    else
        out = tanh(in);
    end
end

function [next_s, next_p, acc_body] = physics_step(curr_s, curr_p, curr_c, tf, tr, dr, Iz, p)
    vx = curr_s(1); vy = curr_s(2); r = curr_s(3); 
    X = curr_p(1); Y = curr_p(2); psi = curr_p(3); 
    
    T = curr_c(1); brake = curr_c(2); delta = curr_c(3);
    
    Frx = (dr(1) - dr(2)*vx)*T - dr(5) * brake - dr(3) - dr(4)*vx^2;
    af = delta - atan2(r*p.lf + vy, vx + 1e-6) + tf(5); 
    ar = atan2(r*p.lr - vy, vx + 1e-6) + tr(5);
    
    Ffy = tf(6) + tf(3)*sin(tf(2)*atan(tf(1)*af - tf(4)*(tf(1)*af - atan(tf(1)*af))));
    Fry = tr(6) + tr(3)*sin(tr(2)*atan(tr(1)*ar - tr(4)*(tr(1)*ar - atan(tr(1)*ar))));
    
    vx_dot = (- Ffy*sin(delta) + Frx)/p.m + vy*r; 
    vy_dot = (  Ffy*cos(delta) + Fry)/p.m - vx*r;
    r_dot = (Ffy*cos(delta)*p.lf - Fry*p.lr) / Iz;
    
    next_s = [vx + vx_dot*p.dt, vy + vy_dot*p.dt, r + r_dot*p.dt];
    next_p = [X + (vx*cos(psi) - vy*sin(psi))*p.dt, Y + (vx*sin(psi) + vy*cos(psi))*p.dt, psi + r*p.dt];
    acc_body = [vx_dot - vy*r, vy_dot + vx*r];
end