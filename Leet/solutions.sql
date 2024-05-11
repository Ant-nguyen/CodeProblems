-- Leet code 175. Combine Two tables
select p.firstName, p.lastname,a.city, a.state
from Person as p 
left outer join Address as a
on (p.personId = a.personId)


-- or
select firstName, lastname, city, state
from Person
left outer join Address 
on (Person.personId = Address.personId)


-- Leet code 182. Duplicate email
select email 
from Person 
group by email
having count(*) > 1