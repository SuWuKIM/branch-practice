# if문 1개로 fizzbuzz 구현하기
for i in range(1, 20+1):
    if i % 3 == 0 or i % 5 == 0:
        print('Fizz' * (i % 3 == 0) + 'Buzz' * (i % 5 == 0))
    else:
        print(f'{i}')
