#include "network.cpp"

class bTrainer {
    private:
        ai::network* model;
    public:
        void genModels() {
            std::vector<int> hiddenLayers {50,50,50,25,10};
            std::vector<std::string> outIds {"in","out","circle","above","cross"};
            this->model = (new ai::network)->genRandNet(2,&hiddenLayers,outIds.size(),&outIds);
        }
        void train(int trainNum) {
            int batchNum = 25;
            std::vector<std::map<std::string,double>> outputs;
            std::vector<std::string> correctOut;
            std::vector<std::vector<std::vector<double>>> layerOuts;
            std::map<std::string,int> dataCnt;
            double lossRate =0;
            for(int i=0;i<batchNum;) {
                std::vector<double> data;
                std::string tag = this->genTrainData(&data);
                if(dataCnt[tag]==5) {
                    continue;
                }
                dataCnt[tag]++;
                i++;
                std::vector<std::vector<double>> layerO;
                std::vector<double>* out = this->model->runNetwork(&data,NULL,&layerO,false);
                std::map<std::string,double> outMap;
                std::vector<double> pOut = *out;
                std::map<std::string,double> pOutMap;
                this->model->outputsListToMap(out,&outMap);
                //this->model->outputsToPercentage(&pOut);
                //this->model->outputsListToMap(&pOut,&pOutMap);
                lossRate += this->model->getLossRate(tag,&outMap);
                //std::cout<<"d: "<<data[0]<<" "<<data[1]<<" "<<outMap[tag]<<" "<<tag<<" "<<(*std::max_element(outMap.begin(),outMap.end(),[](const auto &a,const auto &b){return a.second<b.second;})).first<<std::endl;
                outputs.push_back(outMap);
                correctOut.push_back(tag);
                layerOuts.push_back(layerO);
                delete out;
            }
            std::vector<std::vector<double>> avgLayerOut;
            ai::deltaAndCost dnc =  this->model->getDeltaAndCost(&outputs,&layerOuts,&correctOut);
            if(dnc.deltas.size()){
                this->model->getAvgLayerOutput(&layerOuts,&avgLayerOut);
                this->model->backwardAdjust(&avgLayerOut,&dnc.deltas,&dnc.costs);
            }
            std::cout<<"[Trained "<<trainNum<<"] Result LossRate: "<<lossRate/batchNum<<std::endl;
        }
        std::string genTrainData(std::vector<double>* data) {
            (*data) = *(new std::vector<double> {ai::randRange(-3,3),ai::randRange(-3,3)});
            double x = (*data)[0];
            double y = (*data)[1];
            return getAns(x,y);
        }
        std::string getAns(double x,double y) {
            return x*x*y*y<.1 ? "in" : (x*x+y*y<5.5&&x*x+y*y>3?"circle":(y>1?"above":(y-x>0?"cross":"out")));
        }
        ai::network* getModel() {
            return this->model;
        }
};

int main() {
    srand((unsigned)time(NULL));
    bTrainer train;
    train.genModels();
    int trained = 0;
    int trainNum = 0;
    train.getModel()->importModel("graphModel.txt");
    while(true) {
        for(int i=trained;i<trained+trainNum;i++) {
            train.train(i+1);
            //train.mutate(200);
        }
        trained += trainNum;
        std::cout <<"\n[>]Testing Model... "<<std::flush;
        ai::network* model = train.getModel();
        model->exportModel("graphModel.txt");
        bool notConfiguredPlot = true;
        double tx =-3,ty =-3;
        for(int i=0;ty<=3;i++) {
            std::vector<double> data {tx,ty};
            std::string tag = train.getAns(tx,ty);
            std::map<std::string,double> output;
            model->runNetwork(&data,&output);
            auto maxRes = (*std::max_element(output.begin(),output.end(),[](const auto &a,const auto &b){return a.second<b.second;}));
            std::vector<double> x {data[0]};
            std::vector<double> y {data[1]};
            std::vector<std::string> color {maxRes.first=="in"?"green":(maxRes.first=="circle"?"yellow":(maxRes.first=="above"?"blue":(maxRes.first=="cross"?"purple":"red")))};
            plt::subplot(1,2,1);
            if(notConfiguredPlot)plt::title("AI result");
            plt::scatter_colored(x,y,color);
            plt::subplot(1,2,2);
            if(notConfiguredPlot)plt::title("Real result");
            color[0] = {tag=="in"?"green":(tag=="circle"?"yellow":(tag=="above"?"blue":(tag=="cross"?"purple":"red")))};
            plt::scatter_colored(x,y,color);
            notConfiguredPlot = false;
            tx += 0.1;
            if(tx>=3) {
                tx = -3;
                ty += 0.1;
                std::cout<<"\r[>]Testing Model... "<<std::round(((ty+3)/6)*100)<<'%'<<std::flush;
            }
        }
        plt::show();
        std::cout << "\nNext Train Num: ";
        std::string trainNumS;
        std::getline(std::cin, trainNumS);
        std::cout << "Learning Rate: ";
        std::string learningRate;
        std::getline(std::cin, learningRate);
        if(!trainNumS.empty()) trainNum = std::stoi(trainNumS);
        if(!learningRate.empty()) model->learningRate = std::stod(learningRate);
    }
    
    /*std::vector<int> hiddenLayers {3,3};
    std::vector<std::string> outIds {"in","out"};
    network net;
    net.genRandNet(2,&hiddenLayers,2,&outIds);
    network* net1 = new network;
    *net1 = net;
    std::cout << (*(net1->getLayer()))[0][0].weightsToPrevLayer[0] << " ";
    net1->mutate(1);
    std::cout << (*(net1->getLayer()))[0][0].weightsToPrevLayer[0] << " " << (*net.getLayer())[0][0].weightsToPrevLayer[0] << std::endl;
    
    std::vector<double> inputs {2,1};
    std::map<std::string,double> outputs;
    net.runNetwork(&inputs,&outputs);
    std::cout << round(outputs["in"]*100) <<" "<< net.getLossRate("out",&outputs) << std::endl;
    net1->runNetwork(&inputs,&outputs);
    std::cout << round(outputs["in"]*100) <<" "<< net1->getLossRate("out",&outputs) << std::endl;
    delete net1;*/
    return 0;
}