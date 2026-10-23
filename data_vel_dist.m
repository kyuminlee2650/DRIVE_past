function comprehensive_data_analysis(fs_title, fs_label, fs_tick, fs_legend, fs_text)
    % ---------------------------------------------------------------------
    % 1. 입력 파라미터 기본값 설정 (값 입력이 없을 시 기본으로 키운 크기 적용)
    % ---------------------------------------------------------------------
    if nargin < 1 || isempty(fs_title),  fs_title = 20;  end % 메인 타이틀
    if nargin < 2 || isempty(fs_label),  fs_label = 18;  end % X/Y 축 레이블 (서브타이틀 포함)
    if nargin < 3 || isempty(fs_tick),   fs_tick = 16;   end % 축 틱 레이블 (숫자)
    if nargin < 4 || isempty(fs_legend), fs_legend = 16; end % 범례 (Legend)
    if nargin < 5 || isempty(fs_text),   fs_text = 16;   end % 그래프 내부 텍스트 및 데이터 라벨

    % ---------------------------------------------------------------------
    % 2. Persistent (지속) 변수 선언 및 초기화
    % ---------------------------------------------------------------------
    persistent initialized folders labels group_names source_labels colors
    persistent c_vx c_vy c_thr c_brk fn
    persistent data_vx data_vy data_thr data_brk
    persistent ratio_results std_vx std_vy avg_v max_v
    
    if isempty(initialized)
        initialized = true;
        
        % 폴더 및 라벨 설정
        folders = {
            "C:\workspace\carsim_processed_data_25hz", ...       % 1: Sim Train
            "C:\workspace\carsim_processed_data_test_25hz", ...  % 2: Sim Test
            "C:\workspace\processed_data_25hz", ...              % 3: Real Train
            "C:\workspace\test_csv"                              % 4: Real Test
        };
        labels = {'Sim Train', 'Sim Test', 'Real Train', 'Real Test'};
        group_names = {'Train Datasets', 'Test Datasets'};
        source_labels = {'Simulation data', 'Real-world data'};
        colors = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E'};
        
        % 데이터 열(Column) 인덱스
        c_vx = 5;  % 종방향 속도
        c_vy = 6;  % 횡방향 속도
        c_thr = 8; % 스로틀
        c_brk = 10;% 브레이크
        
        fn = 'Times New Roman'; % 글꼴 종류 고정
        
        % 데이터 저장을 위한 Cell Array 초기화
        data_vx = cell(1, 4);
        data_vy = cell(1, 4);
        data_thr = cell(1, 4);
        data_brk = cell(1, 4);
        
        % 통계량 배열 초기화
        ratio_results = zeros(4, 2);
        std_vx = zeros(1, 4);
        std_vy = zeros(1, 4);
        avg_v = zeros(1, 4);
        max_v = zeros(1, 4);
    end
    
    close all; clc;
    fprintf('데이터 분석을 시작합니다...\n');
    
    % ---------------------------------------------------------------------
    % 3. 데이터 로드 및 전처리 (최초 1회만 파일 리딩)
    % ---------------------------------------------------------------------
    for d = 1:4
        if isempty(data_vx{d}) % persistent 변수가 비어있을 때만 파일 읽기
            fileList = dir(fullfile(folders{d}, '**', '*.csv'));
            tmp_vx = []; tmp_vy = []; tmp_thr = []; tmp_brk = [];
            
            for i = 1:length(fileList)
                try
                    data = readmatrix(fullfile(fileList(i).folder, fileList(i).name));
                    
                    % 속도 데이터 추출 (NaN 제외)
                    valid_v = ~isnan(data(:, c_vx)) & ~isnan(data(:, c_vy));
                    tmp_vx = [tmp_vx; data(valid_v, c_vx)];
                    tmp_vy = [tmp_vy; data(valid_v, c_vy)];
                    
                    % 페달 데이터 추출 (NaN 제외)
                    valid_p = ~isnan(data(:, c_thr)) & ~isnan(data(:, c_brk));
                    tmp_thr = [tmp_thr; data(valid_p, c_thr)];
                    tmp_brk = [tmp_brk; data(valid_p, c_brk)];
                catch
                    fprintf('파일 읽기 실패: %s\n', fileList(i).name);
                end
            end
            % 데이터를 persistent 변수에 저장
            data_vx{d} = tmp_vx; data_vy{d} = tmp_vy;
            data_thr{d} = tmp_thr; data_brk{d} = tmp_brk;
        end
    end
    
    % ---------------------------------------------------------------------
    % 4. 통계 연산
    % ---------------------------------------------------------------------
    for d = 1:4
        % 비율 계산
        total_rows = length(data_thr{d});
        if total_rows > 0
            ratio_results(d, 1) = (sum(data_thr{d} > 0) / total_rows) * 100;
            ratio_results(d, 2) = (sum(data_brk{d} > 0) / total_rows) * 100;
        end
        
        % 분포 및 Excitation 지표 계산
        if ~isempty(data_vx{d})
            std_vx(d) = std(data_vx{d});
            std_vy(d) = std(data_vy{d});
            avg_v(d) = mean(data_vx{d});
            max_v(d) = max(data_vx{d});
        end
    end
    
    % ---------------------------------------------------------------------
    % 5. Figure 1: 속도 분포 비교 (Train / Test)
    % ---------------------------------------------------------------------
    figure('Color', 'w', 'Position', [50, 100, 1100, 550], 'Name', 'Analysis 1: Velocity Distribution');
    t1 = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    
    dist_groups = {[1, 3], [2, 4]}; 
    
    for g = 1:2
        nexttile(t1); hold on; grid on; box on;
        
        current_indices = dist_groups{g};
        h_plots = [];
        text_y_pos = 0.95;
        
        for idx = 1:2
            d = current_indices(idx);
            
            h = histogram(data_vx{d}, 50, 'Normalization', 'probability', ...
                'FaceColor', colors{idx}, 'EdgeAlpha', 0.1, 'FaceAlpha', 0.5, ...
                'DisplayName', source_labels{idx});
            h_plots(idx) = h;
            
            % 통계 텍스트 우측 정렬
            str = sprintf('%s: Avg=%.2f, Max=%.2f', source_labels{idx}, avg_v(d), max_v(d));
            text(0.95, text_y_pos, str, 'Units', 'normalized', 'FontName', fn, ...
                'Color', colors{idx}, 'FontWeight', 'bold', 'FontSize', fs_text, ...
                'HorizontalAlignment', 'right');
            text_y_pos = text_y_pos - 0.08;
        end
        
        set(gca, 'FontName', fn, 'FontSize', fs_tick);
        title(['Velocity Distribution: ', group_names{g}], 'FontName', fn, 'FontSize', fs_title);
        xlabel('Velocity [m/s]', 'FontName', fn, 'FontSize', fs_label);
        ylabel('Probability Density', 'FontName', fn, 'FontSize', fs_label);
    end
    
    lgd1 = legend(h_plots, source_labels);
    lgd1.Orientation = 'horizontal'; lgd1.FontName = fn; lgd1.FontSize = fs_legend;
    lgd1.Layout.Tile = 'south'; 
    
    % ---------------------------------------------------------------------
    % 6. Figure 2: 페달(스로틀/브레이크) 작동 시간 비율
    % ---------------------------------------------------------------------
    figure('Color', 'w', 'Position', [100, 150, 800, 500], 'Name', 'Analysis 2: Pedal Active Ratio');
    b2 = bar(ratio_results, 'grouped');
    b2(1).FaceColor = colors{1}; 
    b2(2).FaceColor = colors{2}; 
    
    grid on;
    set(gca, 'XTickLabel', labels, 'FontName', fn, 'FontSize', fs_tick);
    title('Active Time Ratio: Throttle vs Brake', 'FontName', fn, 'FontSize', fs_title);
    ylabel('Active Time Ratio [%]', 'FontName', fn, 'FontSize', fs_label);
    ylim([0, max(ratio_results, [], 'all') * 1.2]);
    
    lgd2 = legend('Throttle ($>0$)', 'Brake ($>0$)', 'Orientation', 'horizontal', 'Location', 'southoutside', 'Interpreter', 'latex');
    lgd2.FontName = fn; lgd2.FontSize = fs_legend;
    
    for i = 1:2
        xtips = b2(i).XEndPoints;
        ytips = b2(i).YEndPoints;
        labels_txt = string(round(b2(i).YData, 1)) + "%";
        text(xtips, ytips, labels_txt, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontName', fn, 'FontSize', fs_text, 'FontWeight', 'bold');
    end
    
    % ---------------------------------------------------------------------
    % 7. Figure 3: 속도 Excitation (상태 공간 커버리지 산점도)
    % ---------------------------------------------------------------------
    figure('Color', 'w', 'Position', [150, 200, 1000, 800], 'Name', 'Analysis 3-1: State Space Coverage');
    t3 = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    
    for d = 1:4
        nexttile(t3); hold on; grid on; box on;
        
        scatter(data_vx{d}, data_vy{d}, 15, 'filled', ...
            'MarkerFaceColor', colors{d}, 'MarkerEdgeColor', colors{d}, ...
            'MarkerFaceAlpha', 0.15, 'MarkerEdgeAlpha', 0.15);
        
        set(gca, 'FontName', fn, 'FontSize', fs_tick);
        title(labels{d}, 'FontName', fn, 'FontSize', fs_label);
        xlabel('Longitudinal Velocity $v_x$ [m/s]', 'Interpreter', 'latex', 'FontName', fn, 'FontSize', fs_label);
        ylabel('Lateral Velocity $v_y$ [m/s]', 'Interpreter', 'latex', 'FontName', fn, 'FontSize', fs_label);
        
        xline(0, 'k--', 'LineWidth', 1);
        yline(0, 'k--', 'LineWidth', 1);
    end
    title(t3, 'State Space Coverage ($v_x$ vs $v_y$)', 'Interpreter', 'latex', 'FontName', fn, 'FontSize', fs_title, 'FontWeight', 'bold');
    
    % ---------------------------------------------------------------------
    % 8. Figure 4: 속도 Excitation 정량화 (표준편차)
    % ---------------------------------------------------------------------
    figure('Color', 'w', 'Position', [200, 250, 800, 450], 'Name', 'Analysis 3-2: Excitation Metrics');
  
    b4 = bar([std_vx; std_vy]', 'grouped');
    b4(1).FaceColor = colors{1}; 
    b4(2).FaceColor = colors{2}; 
    
    grid on;
    set(gca, 'XTickLabel', labels, 'FontName', fn, 'FontSize', fs_tick);
    title('Velocity Excitation Metrics (Standard Deviation)', 'FontName', fn, 'FontSize', fs_title);
    ylabel('Standard Deviation $\sigma$', 'Interpreter', 'latex', 'FontName', fn, 'FontSize', fs_label);
    
    lgd4 = legend('$\sigma(v_x)$', '$\sigma(v_y)$', 'Interpreter', 'latex', 'Location', 'northwest');
    lgd4.FontName = fn; lgd4.FontSize = fs_legend;
    
    for i = 1:2
        xtips = b4(i).XEndPoints;
        ytips = b4(i).YEndPoints;
        labels_txt = string(round(b4(i).YData, 2));
        text(xtips, ytips, labels_txt, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontName', fn, 'FontSize', fs_text, 'FontWeight', 'bold');
    end
    ylim([0,17]);
    fprintf('\n분석 및 시각화가 완료되었습니다!\n');
end