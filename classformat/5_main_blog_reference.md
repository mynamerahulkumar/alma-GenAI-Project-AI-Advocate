Getting Started with Authentication, authorization, and access control
Introduction 
In today's interconnected and data-driven world, organizations must implement robust mechanisms to authenticate users, authorize their actions, and enforce appropriate access controls. 
Google, a multinational technology company, places great emphasis on authentication, authorization, and access control to protect user data and secure its extensive range of services. With millions of users accessing various Google platforms and applications, the company understands the criticality of ensuring the privacy and security of user accounts and information.
Google's authentication framework includes a combination of factors, such as passwords, two-factor authentication (2FA), and biometric authentication, to verify users' identities. By offering multiple layers of authentication, Google enhances the security of user accounts and reduces the risk of unauthorized access.
notion image
In terms of authorization, Google implements granular access control policies. Users are assigned specific roles and permissions based on their needs and responsibilities. This ensures that individuals have access only to the resources required to perform their tasks while maintaining the principle of least privilege.
Access control at Google extends beyond user accounts to its cloud services and APIs. The company employs robust identity and access management (IAM) solutions to manage and control access to its cloud infrastructure, data storage, and other services. Role-based access control (RBAC) is employed to define and enforce fine-grained access privileges for different user roles and groups.
Google's commitment to authentication, authorization, and access control is driven by its dedication to protecting user privacy and data security. By implementing robust mechanisms and continuously enhancing its security practices, Google sets an example for other organizations in the importance of maintaining strong authentication, authorization, and access control measures to safeguard sensitive information.
Authentication
notion image
In authentication process, identities of the users are verified. Most of the time this verification process includes a username and a password but other methods such as PIN number, fingerprint scan, smart card and such are adapted as well. In order to conduct the process of authentication, it is essential that the user has an account in the system so that the authentication mechanism can 
interrogate that account. Or an account has to be created during the process.
A user is either who they claim to be or someone else. Thus the output of the authentication process is either a yes or no. ‘Maybe’ is treated as a no for security concerns. In addition, the ‘user’ may not be an actual person but an application trying to use a web services API.
Authentication technologies are mainly used with two types of authorization processes: Two factor authentication Multi-factor authentication In the past, multi-factor authentication was vastly popular but due to its difficulties in use, password authentication prevailed.
Two factor authentication, on the other hand, is still a widely used security process that involves two methods of verification. One of them is password verification most of the time. 
Frequently used types of authentication technology are username/password, one-time password and biometric authentication.
Authorization
In authorization process, it is established if the user (who is already authenticated) is allowed to 
have access to a resource. In other words, authorization determines what a user is and is not permitted to do. The level of authorization that is to be given to a user is determined by the metadata concerning the user’s account. Such data can indicate if the user is a member of the 
‘Administrators’ or ‘Customers,’ or it can indicate if the user has paid-subscription for some content.
The processes of authorization also encompass Authorization Management which denotes creating authorization rules. For instance, an administrator can be allowed to create such a rule that lets another user to publish content to a web page. 
We create authorization policies while using social media: Facebook, LinkedIn, Twitter or Instagram have millions of users but we can authorize (to an extent) which of those users can interact with us. Authorization technologies empowers 
businesses by enabling them to control what employees can access, or where and on which device they can access data.
A little level of regulation allows businesses to make sure that their staff can access sensitive data on a secure device operating within the company’s firewall.
notion image

Authentication verifies your identity and authentication enables authorization. An authorization policy dictates what your identity is allowed to do. For example, any customer of a bank can create and use an identity (e.g., a user name) to log into that bank's online service but the bank's authorization policy must ensure that only you are authorized to access your individual 
account online once your identity is verified.
Authorization can be applied to more granular levels than simply a web site or company intranet. Your individual identity can be included in a group of identities that share a common authorization 
policy. For example, imagine a database that contains both customer purchases and a customer's personal and credit card information. 
A merchant could create an authorization policy for this database to allow a marketing group access to all customer purchases but prevent access to all customer personal and credit card information, so that the marketing group could identify popular products to promote or put on sale.
Access Control
In the process of access control, the required security for a particular resource is enforced. Once we establish who the user is and what they can access to, we need to actively prevent that user from accessing anything they should not. Thus we can see access control as the merger of authentication and authorization plus some additional measures like IP-based restrictions. 
notion image
Most of the time security vulnerabilities in applications stem from inadequate access control mechanisms instead of faulty authentication or authorization mechanisms. The reason why is that 
access control is more complex and intricate than other two. Main types of access control are DAC (discretionary access control), RBAC (role-based access control), ABAC (attribute based access control) and MAC (mandatory access control).
Whereas authorization policies define what an individual identity or group may access, access controls – also called permissions or privileges – are the methods we use to enforce such policies. Let's look at examples:
Through Facebook settings – Who can see my stuff? Who can contact me? Who can look me up? – We
allow or deny access to what we post on Facebook to users or the public.
Google Docs settings let us set edit or sharing privileges for documents we use collaboratively.
Flickr settings allow us to create or share albums or images with family, friends, or publicly, under different (e.g., Creative Commons) publishing rights licenses.
Shares and permissions on the MacOS or the Security tab in a Windows OS file properties dialog box allow you to set access privileges for individual files or folders.