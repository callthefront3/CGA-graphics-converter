[Photo to CGA graphics 변환기]

origin 폴더에 있는 사진을
흑백 floyd 변환하여 gray 폴더에 저장
CGA atkinson 변환하여 cga 폴더에 저장


main1.py : 흑백으로 변환
GRAY 컬러로 변환 후 (shape y * x * 1)
floyd or atkinson 알고리즘 적용

★ main2.py :  컬러로 변환 (깔끔하게 평평해짐) 
floyd or atkinson 알고리즘 적용 후 
24-bit colors에서 4-bit RGBI (CGA color)로 변환

main3.py :  컬러로 변환 (깨질 확률 높음, 변환 시간 多)
floyd or atkinson 알고리즘 내에서 
24-bit colors에서 4-bit RGBI (CGA color)로 변환/오차 분산

main4.py :  컬러로 변환 (사실적이고 지저분한 푸른끼)
floyd or atkinson 알고리즘 적용 후 
hsv로 변환 후
hsv를 CGA color로 변환


=> 흑백 변환한다면 main1.py의 floyd 알고리즘 사용
=> 컬러 변환한다면 main2.py의 atkinson 알고리즘 사용

