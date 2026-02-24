%% Code to extract the RECM_LBP
clear all
clc;
[data, sequence]= fastaread('combined/independent_dataset_combined.fasta');

Total_Seq_train=size(sequence,2);
for i=1:(Total_Seq_train)
     i
    SEQ=sequence(i);
    SEQ=cell2mat(SEQ);
    RECM_T = RECMT(SEQ);
    RECM_T=RECM_T';
    % P = uint8(255 * mat2gray(RECM_T));
     % Target size before DCT
    target_rows = 10;
    target_cols = 10;

    % Get actual size
    [r, c] = size(RECM_T);

    % Pad rows and columns if needed
    if r < target_rows || c < target_cols
        padded_RECM_T = zeros(target_rows, target_cols);
        padded_RECM_T(1:r, 1:c) = RECM_T;
        RECM_T = padded_RECM_T;
    end

    % Normalize and convert to 8-bit image range
    P = uint8(255 * mat2gray(RECM_T));

    % Apply 2D DCT
    FF = dct2(P);

    % Extract top-left 10×10 block
    FF = FF(1:10, 1:10);

    % Flatten into feature vector
    RECM_DCT_features(i,:) = FF(:);
    
%%%%%%%%%%% RECM_DCT %%%%%%%%%%%%%%%%
     
    % FF=dct2(P);%matlab function
    % FF=FF(1:10,1:10);
	% RECM_DCT_features(i,:)=FF(:);
    % 
     
    
end

%%%%%%%%%%%%%%%%%%%%%%%% SAVE FILES %%%%%%%%%%%%%%%%%%%%%%%%%
RECM_DCT_features_ACP=[RECM_DCT_features];

% save RECM_DCT_features_ACP RECM_DCT_features_ACP;

%%%% To Create CSV sheet for the data %%%%%%%%%
   
csvwrite('combined/IND_RECM_DCT.csv',RECM_DCT_features_ACP);

