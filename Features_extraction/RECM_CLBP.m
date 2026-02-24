%% Code to extract the RECM_LBP
clear all
clc;
% data = load('RCEM.mat');
% disp(fieldnames(data));  % Shows variable names
% RECM = data.RECM;        % Only if it exists

[data, sequence]= fastaread('combined/independent_dataset_combined.fasta');

Total_Seq_train=size(sequence,2);

for i=1:(Total_Seq_train)
     i
    SEQ=sequence(i);
    SEQ=cell2mat(SEQ);
    RECM_T = RECMT(SEQ);
    RECM_T=RECM_T';
    P = uint8(255 * mat2gray(RECM_T));
    
    
    %%%%%%%%%%% RECM-CLBP %%%%%%%%%%%%%%%%
     FF=clbp(P);
     RECM_CLBP_features(i,:)=FF;
   end

%%%%%%%%%%%%%%%%%%%%%%%% SAVE FILES %%%%%%%%%%%%%%%%%%%%%%%%%
RECM_CLBP_features_ACP=[RECM_CLBP_features];

% save TRN_RECM_CLBP_features RECM_CLBP_features_ACP;

%%%% To Create CSV sheet for the data %%%%%%%%%
   
csvwrite('combined/IND_RECM_CLBP_features.csv',RECM_CLBP_features_ACP);


