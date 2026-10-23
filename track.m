
%% =========================================================================
% Vehicle Trajectory to Track Generation
% 주행 궤적을 기반으로 가상의 트랙(폭)을 생성하고 시각화하는 코드
% =========================================================================
clear all; close all; clc;

%% 1. 데이터 로드 및 파라미터 설정
% 정제된 CSV 파일 경로 지정 (헤더가 없다면 NumHeaderLines 제거)

csv_path = "C:\workspace\carsim_processed_data_test\v20_nooffset1p5sec_new.csv";
% csv_path = "C:\workspace\carsim_processed_data\v20_nooffset1sec_handling.csv";
% csv_path = "C:\workspace\test_csv\20130223_01_02_03_grandsport_25hz.csv"; 
% csv_path = "C:\workspace\processed_data_25hz\20130811_01_01_01_grandsport_25hz.csv"; 
data = readmatrix(csv_path);

%carsim test
X_raw = data(1:26400, 2);
Y_raw = data(1:26400, 3);

%carsim train
% X_raw = data(:, 2);
% Y_raw = data(:, 3);

% targa six
% X_raw = data(1200:5000, 2);
% Y_raw = data(1200:5000, 3);

% 2013 Monterey Motorsports Reunion
% X_raw = data(1500:4850, 2);
% Y_raw = data(1500:4850, 3);

track_width = 20;           
half_width = track_width / 2;
smooth_window = 30;         

%% 2. 궤적 스무딩
X = movmean(X_raw, smooth_window);
Y = movmean(Y_raw, smooth_window);

%% 3. 진행 방향 및 수직 방향 벡터 계산
dX = gradient(X);
dY = gradient(Y);
mag = sqrt(dX.^2 + dY.^2) + 1e-6;
tX = dX ./ mag;
tY = dY ./ mag;
nX = -tY;
nY = tX;

%% 4. 좌우 트랙 경계 좌표 계산
X_left  = X + nX * half_width;
Y_left  = Y + nY * half_width;
X_right = X - nX * half_width;
Y_right = Y - nY * half_width;

%% 5. 플롯 및 Times New Roman 설정
figure('Color', 'w', 'Name', 'Track Visualization', 'Position', [100, 100, 900, 700]);
hold on; axis equal; 

% 트랙 좌우 경계선 그리기
plot(X_left, Y_left, 'k-', 'LineWidth', 4.0, 'DisplayName', 'Track Boundary');
plot(X_right, Y_right, 'k-', 'LineWidth', 4.0, 'DisplayName', 'Track Boundary');

% 글꼴 및 축 설정
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'FontName', 'Times New Roman');

% 타이틀 및 라벨 설정 (FontName 명시)
hTitle = title('Thunderhill Raceway Park (5-mile course)', 'FontSize', 30, 'FontWeight', 'bold');
hXlabel = xlabel('Global X [m]', 'FontSize', 14);
hYlabel = ylabel('Global Y [m]', 'FontSize', 14);

set(hTitle, 'FontName', 'Times New Roman');
set(hXlabel, 'FontName', 'Times New Roman');
set(hYlabel, 'FontName', 'Times New Roman');


hold off;