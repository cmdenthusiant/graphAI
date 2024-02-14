//#include "network.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <fstream>
#define WITHOUT_NUMPY true
#include "matplotlibcpp.h"

#define RAND_hMAX RAND_MAX/2

namespace plt = matplotlibcpp;

namespace ai {
    struct neuron {
        std::vector<double> weightsToPrevLayer;
        std::string id;
        double bias;
    };
    struct deltaAndCost {
        std::vector<std::vector<double>> deltas;
        std::vector<std::vector<std::vector<double>>> costs;
    };

    double randRange(double min, double max) {
        return ((double)rand()/RAND_MAX) * (max-min) + min;
    }

    template <typename iterB,typename iterE,typename elemFunc,typename compareV>
    inline bool find_element(iterB iterBegin,iterE iterEnd,elemFunc elementFunc,compareV value) {
        int count = 0;
        for(;iterBegin!=iterEnd;iterBegin++) {
            auto comValue = elementFunc(iterBegin);
            if(value==comValue) return true;
        }
        return false;
    };

    class network {
        private:
            std::vector<std::vector<neuron>> layers;
            int neuronNum;
            int inputNum;

            void randLayer(std::vector<neuron>* pLayer,int prevLayerNum,int currLayerNum,std::vector<std::string>* ids) {
                for(int ci=0;ci<currLayerNum;ci++) {
                    neuron n;
                    if(ids) n.id = (*ids)[ci];
                    for(int pi=0;pi<prevLayerNum;pi++) {
                        n.weightsToPrevLayer.push_back(randRange(-1,1));
                    }
                    n.bias = 0.1;
                    pLayer->push_back(n);
                }
            };
            void randLayer(std::vector<neuron>* pLayer,int prevLayerNum,int currLayerNum) {
                this->randLayer(pLayer,prevLayerNum,currLayerNum,NULL);
            }
            void calcLayer(std::vector<double>* prevLayerRes,int layerNum,std::vector<double>* layerRes) {
                std::vector<neuron> currLayer = (this->layers)[layerNum];
                for(neuron n : currLayer) {
                    double res = n.bias;
                    int weightLen = n.weightsToPrevLayer.size();
                    for(int i=0;i<weightLen;i++) {
                        double prevNeuronRes = (*prevLayerRes)[i];
                        double weight = n.weightsToPrevLayer[i];
                        res += prevNeuronRes * weight;
                    }
                    res = (layerNum + 1 == this->layers.size() ? this->activation_Sigmod(res) : this->activation_Tanh(res));
                    layerRes->push_back(res);
                }
            }
            void expRes(std::vector<double>* outputRes) {
                int len = (*outputRes).size();
                for(int i=0;i<len;i++) {
                    (*outputRes)[i] = std::exp((*outputRes)[i]);
                }
            }
            void normalizeRes(std::vector<double>* outputRes) {
                int len = (*outputRes).size();
                double total = 0;
                for(double v : *outputRes) total += v;
                for(int i=0;i<len;i++) (*outputRes)[i] /= total;
            }
            double activation_Sigmod(double res) {
                return 1.0/(1+std::exp(-res));
            }
            double derivative_Sigmod(double sig) {
                return sig*(1-sig);
            }

            double activation_Tanh(double res) {
                return (2.0/(1+std::exp(-2 * res))) - 1;
            }
            double derivative_Tanh(double t) {
                return 1-(t*t);
            }
            
        public:
            double learningRate = 0.1;
            std::vector<std::string> outputIds;

            network* genRandNet(int inputNum,std::vector<int>* pHiddenLayerNums,int outputNum,std::vector<std::string>* outputIds) {
                this->inputNum = inputNum;
                std::vector<neuron> firstLayer;
                randLayer(&firstLayer,inputNum,(*pHiddenLayerNums)[0]);
                this->neuronNum = firstLayer.size();
                this->layers.push_back(firstLayer);
                for(int i=1;i<(*pHiddenLayerNums).size();i++) {
                    std::vector<neuron> layer;
                    randLayer(&layer,(*pHiddenLayerNums)[i-1],(*pHiddenLayerNums)[i]);
                    this->neuronNum += layer.size();
                    this->layers.push_back(layer);
                }
                std::vector<neuron> lastLayer;
                randLayer(&lastLayer,(*pHiddenLayerNums)[(*pHiddenLayerNums).size()-1],outputNum,outputIds);
                this->neuronNum += lastLayer.size();
                this->layers.push_back(lastLayer);
                this->outputIds = *outputIds;
                return this;
            };
            std::vector<std::vector<neuron>>* getLayer() {
                return &this->layers;
            }
            
            std::vector<double>* runNetwork(std::vector<double>* inputs,std::map<std::string,double>* outputs,std::vector<std::vector<double>>* layerOutputs=NULL,bool toPercentageMap=true) {
                if(!inputNum||inputs->size()!=inputNum) return NULL;
                std::vector<double> lastLayerRes = (*inputs);
                for(int i=0;;i++) {
                    if(layerOutputs) layerOutputs->push_back(lastLayerRes);
                    if(i>=this->layers.size())break;
                    std::vector<double> currLayerRes;
                    this->calcLayer(&lastLayerRes,i,&currLayerRes);
                    lastLayerRes = currLayerRes;
                }
                if(toPercentageMap) {
                    this->outputsToPercentage(&lastLayerRes);
                    this->outputsListToMap(&lastLayerRes,outputs);
                    return NULL;
                }
                std::vector<double>* res = new std::vector<double>;
                *res = lastLayerRes;
                return res;
            }
            void outputsToPercentage(std::vector<double>* outputList) {
                this->expRes(outputList);
                this->normalizeRes(outputList);
            }
            void outputsListToMap(std::vector<double>* outputL,std::map<std::string,double>* outputM) {
                std::vector<neuron> outputLayer = this->layers.back();
                for(int i=0;i<outputLayer.size();i++) {
                    std::string id = outputLayer[i].id;
                    if(id.empty()) id = std::to_string(i);
                    (*outputM)[id] = outputL->at(i);
                }
            }
            double getLossRate(std::string correctTag,std::map<std::string,double>* outputs) {
                return std::abs(outputs->at(correctTag)-1);
            }
            void mutate(double percent=.3,double mutateRate=.5) {
                if(percent>1) percent = 1;
                int numToMutate = ceil(this->neuronNum * percent);
                int selectedNeuron = 0;
                std::map<int,std::map<int,bool>> inListedNeurons;
                while (selectedNeuron!=numToMutate) {
                    int layerNum = randRange(0,this->layers.size()); // floor so not use size()-1
                    int neuronNum = randRange(0,this->layers[layerNum].size());
                    if(inListedNeurons.count(layerNum)&&inListedNeurons[layerNum][neuronNum]) continue;
                    inListedNeurons[layerNum][neuronNum] = true;
                    selectedNeuron++;
                }
                for(auto layerNum : inListedNeurons) {
                    for(auto neuronNum : layerNum.second) {
                        neuron* n = &this->layers[layerNum.first][neuronNum.first];
                        for(int i=0;i<(*n).weightsToPrevLayer.size();i++) (*n).weightsToPrevLayer[i] += randRange(-mutateRate,mutateRate);
                        n->bias += randRange(-mutateRate,mutateRate);
                    }
                    
                }
            }
            deltaAndCost getDeltaAndCost(std::vector<std::map<std::string,double>>* outputs,std::vector<std::vector<std::vector<double>>>* layerOutputs,std::vector<std::string>* correctOutput) {
                std::map<std::string,int> tagToNeuron;
                std::vector<neuron> outputLayer = this->layers.back();
                std::map<std::string,double> outputDiff;
                std::vector<std::vector<double>> totalDelta;
                std::vector<std::vector<std::vector<double>>> totalCost;
                int outputsNum = outputs->size();
                for(int i=0;i!=outputLayer.size();i++) tagToNeuron[outputLayer[i].id] = i;
                for(auto i=0;i<outputsNum;i++) {
                    double corrPercent = outputs->at(i)[correctOutput->at(i)];
                    /*if((*std::max_element(outputs->at(i).begin(),outputs->at(i).end(),[](const auto &a,const auto &b){return a.second<b.second;})).first==correctOutput->at(i)) {
                        continue;
                    }*/
                    std::vector<std::vector<double>> currDeltas;
                    std::vector<std::vector<std::vector<double>>> currCost;
                    currDeltas.push_back(std::vector<double>(outputLayer.size(),0));
                    for(auto res : (*outputs)[i]) {
                        int neuronIndex = tagToNeuron[res.first];
                        double diff = res.second-((*correctOutput)[i]==res.first?1:0);
                        currDeltas[0][neuronIndex] += diff*this->derivative_Sigmod(res.second);
                    }
                    for(int l=this->layers.size()-2;l>=0;l--) {
                        std::vector<double> currLayerD(this->layers[l].size(),0);
                        std::vector<double>& prevLayerD = currDeltas[currDeltas.size()-1];
                        for(int prevn=0;prevn<prevLayerD.size();prevn++) {
                            for(int n=0;n<currLayerD.size();n++) {
                                currLayerD[n] += this->layers[l+1][prevn].weightsToPrevLayer[n] * prevLayerD[prevn];
                            }
                        }
                        for(int n=0;n<currLayerD.size();n++) {
                            currLayerD[n] *= this->derivative_Tanh(layerOutputs->at(i)[l+1][n]);
                        }
                        currDeltas.push_back(currLayerD);
                    }
                    int j=this->layers.size()-1;
                    for(auto layerDelta : currDeltas) {
                        std::vector<std::vector<double>> layerCost(this->layers[j].size());
                        for(int n=0;n<this->layers[j].size();n++) {
                            neuron& neu = this->layers[j][n];
                            double delta = layerDelta[n];
                            layerCost[n].resize(neu.weightsToPrevLayer.size());
                            for(int w=0;w<neu.weightsToPrevLayer.size();w++) {
                                layerCost[n][w] = delta * layerOutputs->at(i)[j][w]; //prev layer output as layeroutputs as all layer's outputs
                            }
                        }
                        currCost.push_back(layerCost);
                        j--;
                    }
                    if(totalCost.size()==0) {//not inited
                        totalDelta = currDeltas;
                        totalCost = currCost;
                    }else for(int l=0;l<currCost.size();l++) {
                        for(int n=0;n<currDeltas[l].size();n++) totalDelta.at(l)[n] += currDeltas[l][n];
                        for(int n=0;n<currCost[l].size();n++) {
                            for(int w=0;w<currCost[l][n].size();w++) totalCost.at(l)[n][w] += currCost[l][n][w];
                        }
                    }
                    
                }
                std::reverse(totalDelta.begin(),totalDelta.end());
                std::reverse(totalCost.begin(),totalCost.end());
                for(int l=0;l<totalCost.size();l++) {
                    for(int n=0;n<totalDelta.at(l).size();n++) totalDelta.at(l)[n] /= outputsNum;
                    for(int n=0;n<totalCost.at(l).size();n++){
                        for(int w=0;w<totalCost.at(l)[n].size();w++) totalCost.at(l)[n][w] /= outputsNum;
                    }
                }
                deltaAndCost dnc;
                dnc.deltas = totalDelta;
                dnc.costs = totalCost;
                return dnc;
            }
            void getAvgLayerOutput(std::vector<std::vector<std::vector<double>>>* layerOutputs,std::vector<std::vector<double>>* avgOutputs) {
                for(int i=0;i<layerOutputs->at(0).size();i++) {
                    std::vector<double> avgo(layerOutputs->at(0)[i].size(),0);
                    for(std::vector<std::vector<double>> layerO : (*layerOutputs)) {
                        for(int j=0;j<layerO[i].size();j++) avgo[j] += layerO[i][j];
                    }
                    avgOutputs->push_back(avgo);
                }
                for(int i=0;i<avgOutputs->size();i++) {
                    for(int j=0;j<(*avgOutputs)[i].size();j++) (*avgOutputs)[i][j] /= layerOutputs->size();
                }
            }
            void backwardAdjust(int layerNum,std::vector<std::vector<double>>* layerOutputs,std::vector<double>* layerDelta,std::vector<std::vector<double>>* layerCost) {
                std::vector<neuron>* adjectLayer = &this->layers[layerNum];
                std::vector<double> prevLayerOut = (*layerOutputs)[layerNum]; //not layerNum-1 as layers has no inputLayer
                int i=0;
                for(auto n=adjectLayer->begin();n!=adjectLayer->end();n++) {
                    (*n).bias -= layerDelta->at(i) * this->learningRate;
                    for(int w=0;w<(*n).weightsToPrevLayer.size();w++) {
                        (*n).weightsToPrevLayer[w] -= layerCost->at(i)[w];
                    }
                    i++;
                }
            }
            void backwardAdjust(std::vector<std::vector<double>>* layerOutputs,std::vector<std::vector<double>>* layersDelta,std::vector<std::vector<std::vector<double>>>* layersCost) {
                for(int i=this->layers.size()-1;i>=0;i--) {
                    this->backwardAdjust(i,layerOutputs,&layersDelta->at(i),&layersCost->at(i));
                }
            }
            void exportModel(std::string filename){
                std::ofstream file(filename);
                file << this->inputNum << ' ';
                int n = this->outputIds.size();
                file << n << '\n';
                for(int i=0;i<n;i++) {
                    file << this->outputIds[i] << ' ';
                }
                n = this->layers.size();
                file << n << '\n';
                for(int i=0;i<n;i++) {
                    auto currl = this->layers[i];
                    int ln = currl.size();
                    file << ln << '\n';
                    for(ai::neuron neu : currl) {
                        int wn = neu.weightsToPrevLayer.size();
                        file << neu.bias << ' ';
                        for(int k=0;k<wn;k++) {
                            file << neu.weightsToPrevLayer[k]<<' ';
                        }
                    }
                }
                file.close();
            }

            void importModel(std::string filename){
                std::ifstream file(filename);
                file >> this->inputNum;
                int n;
                file >> n;
                this->outputIds.clear();
                for(int i=0;i<n;i++) {
                    std::string s;
                    file >> s;
                    outputIds.push_back(s);
                }
                file >> n;
                this->layers.clear();
                int prevn = this->inputNum;
                for(int i=0;i<n;i++) {
                    std::vector<neuron> layer;
                    int ln;
                    file >> ln;
                    for(int j=0;j<ln;j++) {
                        ai::neuron neu;
                        file >> neu.bias;
                        for(int k=0;k<prevn;k++) {
                            double w;
                            file >> w;
                            neu.weightsToPrevLayer.push_back(w);
                        }
                        if(i+1==n) neu.id = outputIds[j];
                        layer.push_back(neu);
                    }
                    prevn = ln;
                    this->layers.push_back(layer);
                }
                file.close();
            }
    };
}
