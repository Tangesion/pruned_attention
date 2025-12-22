namespace PE {

extern void ___register_AddUnitCreator();
extern void ___register_ConvertUnitCreator();  
extern void ___register_MacUnitCreator();
extern void ___register_MultiplyUnitCreator();

void registerComponents() {
    ___register_AddUnitCreator();
    ___register_ConvertUnitCreator();
    ___register_MacUnitCreator();
    ___register_MultiplyUnitCreator();
}

} // namespace PE