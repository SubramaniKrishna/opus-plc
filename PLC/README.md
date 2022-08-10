# Neural PLC

### Main Files
#### Network Training
- opus_plc_lossfuncs.py
- opus_plc_loader.py
- opus_plc_network.py
- opus_plc_training.py

#### Helper Functions
- mdct_funcs.py
- soft_smooth.py

#### Demo Training:              
python ./opus_plc_training.py ~/datasets/tts_speech_48kfull_bandLogE ~/datasets/tts_speech_48kfull_mdctnormalized ~/datasets/train_loss.u8 ./final_weights/shape_model_testing --gru-size 256 --cond-size 256 --epochs 2000 --seq-length 100 --batch-size 32

#### Data Extraction:
Do the following changes to celt_encoder.c
1. Return 0 in the transient_analysis function
2. Add the following lines after line 1743 (or after the amp2log2 function call in celt_encode_with_ec):
   FILE *fp;
   fp = fopen("comp_mono_raw_48k_bandLogE","a");
   for(int z = 0; z < 21; z++){
      // fprintf(fp,"%f",bandLogE[z]);
      fwrite( &bandLogE[z], 1, sizeof(bandLogE[z]),fp ) ;
   }
   // fprintf(fp,"\n");
   fclose(fp);
3. Add the following after line 1880 (or after the normalise_bands function call in celt_encode_with_ec):
FILE *fp1;
   fp1 = fopen("comp_mono_raw_48k_mdctnormalized","a");
   for(int z = 0; z < 960; z++){
      // fprintf(fp1,"%f",X[z]);
      fwrite( &X[z], 1, sizeof(X[z]),fp1 ) ;
   }
   // fprintf(fp1,"\n");
   fclose(fp1);

Recompile and run the opus_demo, it will generate the dump files which can be used for training.

#### Notebook
Check out Combined_Model_PLC.ipynb to analyze the networks outputs and to do PLC.


