clear all; close all;

% 1. 데이터 로드 및 변수 할당
load("C:\workspace\best\DRIVE_carsim.mat")
drive_carsim_loss = val_loss_history;

load("C:\workspace\best\DRIVE_real.mat")
drive_real_loss = val_loss_history;

% 2. 논문용 시각화 스타일 설정
font_name = 'Times New Roman';
title_fs  = 20;  
label_fs  = 18;  
tick_fs   = 16;  
lw_main   = 2.0;  
lw_vline  = 1.2;  

% 그라데이션 색상 정의 (우측으로 갈수록 훨씬 더 짙은 파란색이 되도록 끝 색상 대폭 조정)
color_start = [0.96, 0.98, 1.00]; % 아주 연한 아이스 블루
color_end   = [0.20, 0.40, 0.85]; % 확연히 짙은 딥 블루

% 텍스트 주황색 RGB 설정 (파란 배경에서 잘 보이는 쨍한 주황색)
color_orange = [0.95, 0.40, 0.00]; 

figure('Name', 'DRIVE-Net Validation Loss History', 'NumberTitle', 'off', 'Position', [100, 100, 800, 750]);

%% ========================================================================
% 📈 Subplot 1: 실차 주행 데이터 (Real-world Driving Data)
%% ========================================================================
subplot(2,1,1)
plot(drive_real_loss, 'b-', 'LineWidth', lw_main); 
hold on; grid on; box on;

title('DRIVE-Net Validation Loss (Real-world Data)', 'FontSize', title_fs, 'FontWeight', 'bold', 'FontName', font_name);
xlabel('Epoch', 'FontSize', label_fs, 'FontName', font_name);
ylabel('Validation Loss', 'FontSize', label_fs, 'FontName', font_name);
set(gca, 'FontSize', tick_fs, 'FontName', font_name);

max_epoch1 = length(drive_real_loss);
xlim([1, max_epoch1]);

% 텍스트 삽입을 위해 Y축 상단에 10% 여백 추가
current_ylims1 = ylim;
y_margin1 = (current_ylims1(2) - current_ylims1(1)) * 0.1;
ylim([current_ylims1(1), current_ylims1(2) + y_margin1]);
ylims1 = ylim; 

% [규칙 1] x = 10, 20, 30 ... 구간 나누기
x_lines1 = 10:10:max_epoch1;
intervals1 = [1, x_lines1];
if intervals1(end) < max_epoch1
    intervals1 = [intervals1, max_epoch1];
end
num_int1 = length(intervals1) - 1;

% 구역별 배경색 칠하기 및 텍스트 삽입
for i = 1:num_int1
    x1 = intervals1(i);
    x2 = intervals1(i+1);
    
    % 그라데이션 색상 계산
    t = (i-1) / max(1, num_int1-1);
    curr_color = (1-t)*color_start + t*color_end;
    
    % 배경 패치
    h_patch = patch([x1 x2 x2 x1], [ylims1(1) ylims1(1) ylims1(2) ylims1(2)], curr_color, ...
                    'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(h_patch, 'bottom');
    
    % 텍스트 로직: 첫 칸은 'r = 4', 그 이후는 숫자만
    r_val = 3 + i; 
    if i == 1
        txt_str = sprintf('r = %d', r_val);
    else
        txt_str = sprintf('%d', r_val);
    end
    
    x_center = (x1 + x2) / 2;
    y_pos = ylims1(2) - (ylims1(2) - ylims1(1)) * 0.03; % 상단에서 3% 아래 위치
    
    text(x_center, y_pos, txt_str, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontSize', 14, 'FontName', font_name, 'FontWeight', 'bold', 'Color', color_orange);
end

% 구분 점선 그리기
for x = x_lines1
    if x < max_epoch1
        xline(x, 'k--', 'LineWidth', lw_vline, 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    end
end

set(gca, 'Layer', 'top');

%% ========================================================================
% 📈 Subplot 2: 카심 주행 데이터 (CarSim Simulation Data)
%% ========================================================================
subplot(2,1,2)
plot(drive_carsim_loss, 'r-', 'LineWidth', lw_main); 
hold on; grid on; box on;

title('DRIVE-Net Validation Loss (Simulation Data)', 'FontSize', title_fs, 'FontWeight', 'bold', 'FontName', font_name);
xlabel('Epoch', 'FontSize', label_fs, 'FontName', font_name);
ylabel('Validation Loss', 'FontSize', label_fs, 'FontName', font_name);
set(gca, 'FontSize', tick_fs, 'FontName', font_name);

max_epoch2 = length(drive_carsim_loss);
xlim([1, max_epoch2]);

% 텍스트 삽입을 위해 Y축 상단에 10% 여백 추가
current_ylims2 = ylim;
y_margin2 = (current_ylims2(2) - current_ylims2(1)) * 0.1;
ylim([current_ylims2(1), current_ylims2(2) + y_margin2]);
ylims2 = ylim;

% [규칙 2] x = 25, 45, 60, 70 ... 구간 나누기
x_lines2 = [25, 45, 60:10:max_epoch2];
x_lines2 = unique(x_lines2(x_lines2 <= max_epoch2)); 
intervals2 = [1, x_lines2];
if intervals2(end) < max_epoch2
    intervals2 = [intervals2, max_epoch2];
end
num_int2 = length(intervals2) - 1;

% 구역별 배경색 칠하기 및 텍스트 삽입
for i = 1:num_int2
    x1 = intervals2(i);
    x2 = intervals2(i+1);
    
    t = (i-1) / max(1, num_int2-1);
    curr_color = (1-t)*color_start + t*color_end;
    
    h_patch = patch([x1 x2 x2 x1], [ylims2(1) ylims2(1) ylims2(2) ylims2(2)], curr_color, ...
                    'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(h_patch, 'bottom');
    
    % 텍스트 로직: 첫 칸은 'r = 4', 그 이후는 숫자만
    r_val = 3 + i;
    if i == 1
        txt_str = sprintf('r = %d', r_val);
    else
        txt_str = sprintf('%d', r_val);
    end
    
    x_center = (x1 + x2) / 2;
    y_pos = ylims2(2) - (ylims2(2) - ylims2(1)) * 0.03;
    
    text(x_center, y_pos, txt_str, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontSize', 14, 'FontName', font_name, 'FontWeight', 'bold', 'Color', color_orange);
end

% 구분 점선 그리기
for x = x_lines2
    if x < max_epoch2
        xline(x, 'k--', 'LineWidth', lw_vline, 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    end
end

set(gca, 'Layer', 'top');
