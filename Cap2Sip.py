import numpy as np 
import matplotlib
#matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
from scipy import stats

def MedianFilter(Data,TimeWindow): # TimeWindow has to be EVEN NUMBER!!!
    TotalTime=Data.shape[0] #TotalTime= Real Time/10ms cause the pad samples at 100Hz
    Output=np.zeros(Data.shape)
    for i in range(0,TotalTime):
        Output[i]=np.median(Data[max(0,i-int(TimeWindow/2)):min(TotalTime+1,i+int(TimeWindow/2))],axis=0)
    return(Output)


def MedianNonzero(Data,TimeWindow): # TimeWindow has to be EVEN NUMBER!!! Calculate median on only nonzero values
    TotalTime=Data.shape[0] #TotalTime= Real Time/10ms cause the pad samples at 100Hz
    Channels=Data.shape[1]
    IntervalNum=TotalTime//TimeWindow
    Output=np.zeros(Data.shape)
    TempOutput=np.zeros(Channels)
    for i in range(0,IntervalNum+1):
        a=Data[i*TimeWindow:min(TotalTime+1,(i+1)*TimeWindow)]
        for j in range(0,Channels):
            b=a[:,j]
            TempOutput[j]=np.median(b[np.nonzero(b)])
        Output[i*TimeWindow:min(TotalTime+1,(i+1)*TimeWindow)]=TempOutput
    return(Output)


def LowPassFilter(Data,TimeWindow): # Average in timewindow, unit of the time window is 10ms NOT 1ms!!! TimeWindow has to be EVEN NUMBER!!!
    TotalTime=Data.shape[0] #TotalTime= Real Time/10ms cause the pad samples at 100Hz
    Output=np.zeros(Data.shape)
    for i in range(0,TotalTime):
        Output[i]=np.mean(Data[max(0,i-int(TimeWindow/2)):min(TotalTime+1,i+int(TimeWindow/2))],axis=0)
    return(Output)

def RMS(Data,TimeWindow): # Root mean squared in timewindow, unit of the time window is 10ms NOT 1ms!!! TimeWindow has to be EVEN NUMBER!!!
    TotalTime=Data.shape[0] #TotalTime= Real Time/10ms cause the pad samples at 100Hz
    Output=np.zeros(Data.shape)
    Data2=Data**2 #Data^2
    for i in range(0,TotalTime):
        Output[i]=np.mean(Data2[max(0,i-int(TimeWindow/2)):min(TotalTime+1,i+int(TimeWindow/2))],axis=0)
    Output=np.sqrt(Output)
    return(Output)

def DataProcessor(FileAddress,Median_TW=6,LowPass_TW=50,RMS_TW=50): #Median filter time window default=6; Lowpass filter time window default = RMS time window default=50
    f= open(FileAddress) #read file Captest in binary format
    Cap=np.fromfile(f,'uint16') #Array Cap contains all data
    f.close()
    CapofChannels=Cap.reshape(int(len(Cap)/64),64) #reshape Cap and split it into 64 channels, 2 channels per chamber
    Data_MedianFiltered=MedianFilter(CapofChannels,Median_TW)
    Data_Refiltered=Data_MedianFiltered-LowPassFilter(Data_MedianFiltered,LowPass_TW)
    Data_RMS=RMS(Data_Refiltered,RMS_TW)
    np.save(FileAddress+"_MedianFiltered.npy",Data_MedianFiltered)  
    np.save(FileAddress+"_Refiltered.npy",Data_Refiltered)
    np.save(FileAddress+"_RMS.npy",Data_RMS)
    #print(Data_RMS.shape)
    return(0)

def ChannelPlot(FileAddress,FileType,Channel,PngDPI=1200): 
    FileTypeList={
        0: "_MedianFiltered",
        1: "_Refiltered",
        2: "_RMS",
        3: "_DiffAndThres",
        4: "_Sip_Accum",
        5: "_Sip_Dur"
    }
    InputArray=np.load(FileAddress+FileTypeList[FileType]+".npy")
    x=np.arange(0,InputArray.shape[0]/100,0.01)
    y=InputArray[:,Channel-1]
    plt.plot(x, y) 
    #plt.savefig(FileAddress+FileTypeList[FileType]+"_Channel"+str(Channel)+".png", dpi=PngDPI)
    plt.show()
    return(0)

def DiffAndThres(FileAddress,ThresWindow=100): #diff the matrix and cutoff at a certain threshold
    InputArray=np.load(FileAddress+"_Refiltered"+".npy")
    DiffInput=np.diff(InputArray,axis=0)
    DiffP=DiffInput.copy()
    DiffP[DiffP<=0]=0 
    DiffN=DiffInput.copy()
    DiffN[DiffN>=0]=0
    ThresP=np.nan_to_num(5.926*MedianNonzero(DiffP,ThresWindow))
    ThresN=np.nan_to_num(5.926*MedianNonzero(DiffN,ThresWindow))
    DiffP[np.array(DiffP<=ThresP) | np.array(DiffP<=10)]=0
    DiffN[np.array(DiffN>=ThresN) | np.array(DiffN>=-10)]=0
    DiffOutput=DiffN+DiffP
    np.save(FileAddress+"_DiffAndThres.npy",DiffOutput)
    return(0)

def FindSips(FileAddress): #Annotate sips based on the nc paper
    Diff=np.load(FileAddress+"_DiffAndThres.npy")
    RMS=np.load(FileAddress+"_RMS.npy")
    #Refiltered=np.load(FileAddress+"_Refiltered.npy")
    Channels=Diff.shape[1]
    Pointer=np.zeros([4,Channels]) #Creat a array to store 4 value, including increasing_start, increasing_end, decreasing_start and decreasing_end
    Pointer=Pointer.astype(int)
    Sip_Accum=np.zeros(Diff.shape) #Calculate accumulated sips number
    Sip_Dur=np.zeros(Diff.shape)
    for i in range(1,Diff.shape[0]-1):
        Pointer[np.tile(np.array(Diff[i]<=0) & np.array(Pointer[1]==Pointer[0]),(4,1))]=i 
        Pointer[1:4][np.tile(np.array(Diff[i]>0),(3,1))]=i
        Pointer[0][np.array(Diff[i]>0) & np.array(Diff[i-1]<=0)]=i-1
        Pointer[np.tile(np.array(Diff[i]<0) & np.array(Pointer[1]>Pointer[0]) & np.array(Pointer[1]==Pointer[2]),(4,1))]=i 
        Pointer[1:4][np.tile(np.array(Diff[i]==0) & np.array(Diff[i-1]>0) & np.array(Pointer[1]==Pointer[2]),(3,1))]=i
        Pointer[2:4][np.tile(np.array(Diff[i]==0) & np.array(Diff[i-1]==0) & np.array(Pointer[2]-Pointer[1]<300),(2,1))]=i
        Pointer[np.tile(np.array(Diff[i]==0) & np.array(Diff[i-1]==0) & np.array(Pointer[1]>Pointer[0]) & np.array(Pointer[2]-Pointer[1]==300),(4,1))]=i
        Pointer[3]=i
        # Check if the event end
        Sip_Accum[i]=Sip_Accum[i-1]
        Sip_Dur[i]=Sip_Dur[i-1]
        Z=np.zeros(Channels)
        if i != Diff.shape[0]-1:
            Z[np.array(Pointer[1]-Pointer[0]>0) & np.array(Pointer[1]-Pointer[0]<7) & np.array(Pointer[3]-Pointer[2]>0) & np.array(Pointer[3]-Pointer[2]<7) & np.array(Diff[i+1]>=0)] +=1            
        else:
            Z[np.array(Pointer[1]-Pointer[0]>0) & np.array(Pointer[1]-Pointer[0]<7) & np.array(Pointer[3]-Pointer[2]>0) & np.array(Pointer[3]-Pointer[2]<7)] +=1        
        if len(np.nonzero(Z)[0]) !=0:
            for j in np.nonzero(Z)[0]:
               if RMS[int((Pointer[1,j]+Pointer[2,j])/2),j]>10:
                    #if Refiltered[Pointer[1,j],j]-Refiltered[Pointer[0,j],j]-2*(Refiltered[Pointer[2,j],j]-Refiltered[Pointer[3,j],j])<0:    #Despite of this is one the criteria given by the fypad paper, I found it would introuduce a large false-negative effect.
                    Sip_Accum[i,j] +=1  
                    Sip_Dur[i,j] += 0.01*(Pointer[2,j]-Pointer[1,j])      
        Pointer[np.tile(np.array(Pointer[1]-Pointer[0]>0) & np.array(Pointer[1]-Pointer[0]<7) & np.array(Pointer[3]-Pointer[2]>0) & np.array(Pointer[3]-Pointer[2]<7) & np.array(Diff[i+1]>=0),(4,1))]=i
    Sip_Accum[Diff.shape[0]-1]=Sip_Accum[Diff.shape[0]-2]
    Sip_Dur[Diff.shape[0]-1]=Sip_Dur[Diff.shape[0]-2]
    np.save(FileAddress+"_Sip_Accum.npy",Sip_Accum)
    np.save(FileAddress+"_Sip_Dur.npy",Sip_Dur)
    return(0)

def PlotSip(FileAddress,DurOrAccum,ChannelStart,ChannelEnd,Interval=1,ErrorBand=1): #DurOrAccum must be "Dur" or "Accum"; choose the channel you want to plot starting with and end at; you can choose to plot even channels only by using interval=2; you can set the error bar as x times to S.E.M.
    SipAccum=np.load(FileAddress+"_Sip_"+DurOrAccum+".npy")
    Filter=np.load(FileAddress+"_MedianFiltered.npy")
    DataPlot=SipAccum[:,ChannelStart-1:ChannelEnd:Interval]
    DataPlot=np.delete(DataPlot,np.where(Filter[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0],axis=1) #delete all 4095 which means a broken channel
    print(np.where(Filter[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0])
    MeanSem=np.zeros([DataPlot.shape[0],2])
    MeanSem[:,0]=np.mean(DataPlot,axis=1)
    MeanSem[:,1]=stats.sem(DataPlot,axis=1)
    x=np.arange(0,DataPlot.shape[0]/100,0.01)
    y=MeanSem[:,0]
    ye=MeanSem[:,1]
    plt.plot(x, y,color="#0000FF")
    plt.fill_between(x,y-ErrorBand*ye,y+ErrorBand*ye, color="#9999FF")  
    plt.savefig(FileAddress+"_Sip_"+DurOrAccum+".png", dpi=1200)
    plt.show()
    return(0)

def PlotSip2(Address,FileName1,FileName2,DurOrAccum,ChannelStart,ChannelEnd,Interval=1,ErrorBand=1): #DurOrAccum must be "Dur" or "Accum"; choose the channel you want to plot starting with and end at; you can choose to plot even channels only by using interval=2; you can set the error bar as x times to S.E.M.
    FileAdd1=Address+FileName1
    FileAdd2=Address+FileName2
    
    SipAccum1=np.load(FileAdd1+"_Sip_"+DurOrAccum+".npy")
    Filter1=np.load(FileAdd1+"_MedianFiltered.npy")
    DataPlot1=SipAccum1[:,ChannelStart-1:ChannelEnd:Interval]
    DataPlot1=np.delete(DataPlot1,np.where(Filter1[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0],axis=1)
    print(np.where(Filter1[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0])
    
    SipAccum2=np.load(FileAdd2+"_Sip_"+DurOrAccum+".npy")
    Filter2=np.load(FileAdd2+"_MedianFiltered.npy")
    DataPlot2=SipAccum2[:,ChannelStart-1:ChannelEnd:Interval]
    DataPlot2=np.delete(DataPlot2,np.where(Filter2[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0],axis=1)
    print(np.where(Filter2[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0])

    MeanSem1=np.zeros([DataPlot1.shape[0],2])
    MeanSem1[:,0]=np.mean(DataPlot1,axis=1)
    MeanSem1[:,1]=stats.sem(DataPlot1,axis=1)

    MeanSem2=np.zeros([DataPlot2.shape[0],2])
    MeanSem2[:,0]=np.mean(DataPlot2,axis=1)
    MeanSem2[:,1]=stats.sem(DataPlot2,axis=1)

    x1=np.arange(0,DataPlot1.shape[0]/100,0.01)
    y1=MeanSem1[:,0]
    ye1=MeanSem1[:,1]
    x2=np.arange(0,DataPlot2.shape[0]/100,0.01)
    y2=MeanSem2[:,0]
    ye2=MeanSem2[:,1]
    
    plt.plot(x1, y1,color="#0000FF",label=FileName1)
    plt.fill_between(x1,y1-ErrorBand*ye1,y1+ErrorBand*ye1, color="#9999FF")  
    plt.plot(x2, y2,color="#FF0000",label=FileName2)
    plt.fill_between(x2,y2-ErrorBand*ye2,y2+ErrorBand*ye2, color="#FF9999")
    plt.legend(loc='upper left')
    plt.savefig(FileAdd1+"_vs_"+FileName2+"_Sip_"+DurOrAccum+".png", dpi=1200)
    plt.show()
    return(0)

def RasterPlot(FileAddress,ChannelStart,ChannelEnd,Interval=1,Bin=50, Col="#008748"): #Raster Plot mannually by fill() function
    SipDur=np.load(FileAddress+"_Sip_Dur"+".npy")
    Filter=np.load(FileAddress+"_MedianFiltered.npy")
    DataPlot=SipDur[:,ChannelStart-1:ChannelEnd:Interval]
    DataPlot=np.delete(DataPlot,np.where(Filter[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0],axis=1)
    print(np.where(Filter[-1,ChannelStart-1:ChannelEnd:Interval]>4094.5)[0])
    
    SipTime=np.diff(DataPlot,axis=0)
    for i in range(0,SipTime.shape[1]):
        plt.eventplot(np.nonzero(SipTime[:,i])[0],lineoffsets=i,linewidths=0.1,colors="#000000",)
    plt.savefig(FileAddress+"_SipRasterPlot.png",format='png',dpi=4096)
    plt.show()
  
#Plot eating time ratio within bin seconds
    SipTime=np.zeros([DataPlot.shape[0]+3,1])
    Mean=np.mean(DataPlot,axis=1)
    for i in range(1,len(Mean)+1):
        SipTime[i]=(Mean[min(len(Mean)-1,i-1+Bin*50)]-Mean[max(0,i-1-Bin*50)])/Bin
    x=np.arange(0,len(SipTime)/100,0.01)
    y=SipTime
    plt.fill(x, y,color=Col)
    plt.savefig(FileAddress+"_Sip_Time_Ratio_Within"+str(Bin)+"Seconds.png", dpi=1200)
    plt.show()

    return(0)

def SipChannelPlot(FileAddress,Channel,Bin=50,PngDPI=1200): #bin has a unit of second.
    InputArray=np.load(FileAddress+"_Sip_Dur"+".npy")
    ChannelDur=InputArray[:,Channel-1]
    SipTime=np.zeros([len(ChannelDur),1])
    for i in range(0,len(ChannelDur)):
        SipTime[i]=(ChannelDur[min(len(ChannelDur)-1,i+Bin*50)]-ChannelDur[max(0,i-Bin*50)])/Bin
    x=np.arange(0,len(SipTime)/100,0.01)
    y=SipTime
    plt.plot(x, y,color="#0000FF")
    plt.savefig(FileAddress+"Channel"+str(Channel)+"_Sip_Time_Ratio_Within"+str(Bin)+"Seconds.png", dpi=1200)
    plt.show()
    return(0)


FileAddress='C:/Cap2Sip/' #Please fill it with path of your raw data 
FileName1='Cap20200301T142450'  #Please fill it with file name of your raw data 
FileName2='Cap20200301T145917'  #Please fill it with file name of your raw data 

# These three functions are essential
DataProcessor(FileAddress+FileName1)
DiffAndThres(FileAddress+FileName1)
FindSips(FileAddress+FileName1)
DataProcessor(FileAddress+FileName2)
DiffAndThres(FileAddress+FileName2)
FindSips(FileAddress+FileName2)

# RasterPlot plots eating events of each channel.
RasterPlot(FileAddress+FileName1,1,24,1)
RasterPlot(FileAddress+FileName2,1,24,1)

# PlotSip2 plots a combined figure of two different files(genotypes). It has a Accum mode ploting total recorded sips until the given time; and a Dur mode ploting total time of those reccored sips.
# PlotSip is the single file version of PlotSip2, and it has similar usage.
PlotSip2(FileAddress,FileName1,FileName2,"Accum",1,24,1)
PlotSip2(FileAddress,FileName1,FileName2,"Dur",1,24,1)

# SipChannelPlot plots the ratio of total time of sips in 50s of SINGLE channels(files). 
SipChannelPlot(FileAddress+FileName1,10)

'''
ChannelPlot could plot all intermediate data channel by channel. It has 6 modes listed below.
    FileTypeList={
        0: "_MedianFiltered",
        1: "_Refiltered",
        2: "_RMS",
        3: "_DiffAndThres",
        4: "_Sip_Accum",
        5: "_Sip_Dur"
    }
'''
for i in range(1,25):
    ChannelPlot(FileName+FileName1,3,i) 
